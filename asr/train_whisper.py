"""Fine-tune Whisper on verified Algerian Darija speech.

Default: whisper-large-v3-turbo + LoRA (r=32) on all attention + MLP projections of encoder and
decoder, bf16 base weights, fp32 adapters, gradient checkpointing. Fits the 8 GB RTX 5050.
`--mode full` trains every weight (fp32 master weights + 8-bit AdamW); only use it for
whisper-small/medium, turbo does not fit in 8 GB that way.

Every --eval-every steps: greedy WER/CER on a fixed 300-clip subset of Casablanca-Algeria
validation (human-labelled, never trained on). The best adapter is kept in <run>/best.

  python -m asr.train_whisper --run turbo_lora_v1
  python -m asr.train_whisper --run turbo_lora_v1 --resume      # continue from <run>/last
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.data import DEFAULT_MIX, AudioStore, select_train  # noqa: E402
from asr.eval_asr import score, select_eval, transcribe  # noqa: E402
from common.audio import N_SAMPLES, LogMel  # noqa: E402
from common.paths import HF_CACHE, RUNS  # noqa: E402

def build_train_set(mix, tokenizer, seed=0, max_label_tokens=440):
    recs, report = select_train(mix, seed)
    # tokenise once: <|startoftranscript|><|ar|><|transcribe|><|notimestamps|> text <|endoftext|>
    ids = tokenizer([r["text"] for r in recs], add_special_tokens=True).input_ids
    out = []
    for r, t in zip(recs, ids):
        if len(t) <= max_label_tokens:
            out.append(dict(bin_path=r["bin_path"], off=r["off"], n=r["n"], labels=t[1:]))  # drop SOT; the model prepends it
    return out, report


class ClipDataset(Dataset):
    def __init__(self, recs, train=True):
        self.recs, self.train, self.store = recs, train, None

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        if self.store is None:
            self.store = AudioStore()
        r = self.recs[i]
        a = self.store.get(r)[:N_SAMPLES]
        if self.train:
            a = a * (10 ** (random.uniform(-6, 6) / 20))  # +-6 dB gain
            peak = np.abs(a).max()
            if peak > 1.0: a = a / peak
        return a, r["labels"]


def collate(batch):
    wav = torch.zeros(len(batch), N_SAMPLES)
    L = max(len(l) for _, l in batch)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (a, l) in enumerate(batch):
        wav[i, :len(a)] = torch.from_numpy(a)
        labels[i, :len(l)] = torch.tensor(l)
    return wav, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--mode", choices=["lora", "full"], default="lora")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mix", default=None, help='JSON, e.g. \'{"kahwa": null, "loubna": 60}\'')
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="run N steps, eval on 32 clips, exit")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    from common import winrun
    winrun.apply()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.models.whisper.modeling_whisper import shift_tokens_right

    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_f = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = "cuda"

    processor = WhisperProcessor.from_pretrained(args.base, cache_dir=str(HF_CACHE))
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)

    mix = json.loads(args.mix) if args.mix else DEFAULT_MIX
    train_recs, report = build_train_set(mix, tok)
    total_h = sum(v["hours"] for v in report.values())
    print(f"train set: {len(train_recs)} clips, {total_h:.1f} h  {report}")
    eval_recs = select_eval("casablanca_val", 32 if args.smoke else args.eval_limit)

    if args.mode == "lora":
        model = WhisperForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16, cache_dir=str(HF_CACHE))
    else:
        src = str(run_dir / "last") if args.resume and (run_dir / "last" / "config.json").exists() else args.base
        model = WhisperForConditionalGeneration.from_pretrained(src, dtype=torch.float32, cache_dir=str(HF_CACHE))
    model.config.use_cache = False
    model.config.apply_spec_augment = True       # SpecAugment on the encoder input (train mode only)
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model.config.mask_feature_length = 10
    model.generation_config.forced_decoder_ids = None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.mode == "lora":
        if args.resume and (run_dir / "last" / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(run_dir / "last"), is_trainable=True)
        else:
            cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
            model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    model.to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.mode == "full":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    total_steps = args.smoke or args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    step, best_wer = 0, float("inf")
    state_f = run_dir / "last" / "trainer_state.pt"
    if args.resume and state_f.exists():
        st = torch.load(state_f, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best_wer = st["step"], st["best_wer"]
        print(f"resumed at step {step}, best WER {best_wer:.4f}")

    ds = ClipDataset(train_recs)
    g = torch.Generator(); g.manual_seed(step)  # different order after resume, still reproducible
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0, generator=g,
                    prefetch_factor=4 if args.workers else None)
    logmel = LogMel(processor.feature_extractor, dev)
    pad_id, start_id = model.config.pad_token_id, model.config.decoder_start_token_id

    def evaluate():
        model.eval(); model.config.use_cache = True
        t = time.time()
        hyps = transcribe(model, processor, eval_recs, batch_size=16)
        s = score([r["raw"] for r in eval_recs], hyps)
        model.train(); model.config.use_cache = False
        return s, hyps, time.time() - t

    def save(dirname):
        d = run_dir / dirname
        model.save_pretrained(str(d)); processor.save_pretrained(str(d))
        return d

    model.train()
    it = iter(dl)
    t_last, last_log_step, loss_acc, tokens_acc = time.time(), step, 0.0, 0
    print(f"start: step {step}/{total_steps}, batch {args.batch}x{args.accum}, lr {args.lr}")
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                wav, labels = next(it)
            except StopIteration:
                it = iter(dl); wav, labels = next(it)
            wav, labels = wav.to(dev, non_blocking=True), labels.to(dev, non_blocking=True)
            feats = logmel(wav).to(torch.bfloat16 if args.mode == "lora" else torch.float32)
            dec_in = shift_tokens_right(labels, pad_id, start_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_features=feats, decoder_input_ids=dec_in).logits
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            (loss / args.accum).backward()
            loss_acc += loss.item(); tokens_acc += 1
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1

        if step % 20 == 0 or step == 1 or args.smoke:
            dt = (time.time() - t_last) / max(1, step - last_log_step)
            t_last, last_log_step = time.time(), step
            rec = dict(step=step, loss=round(loss_acc / tokens_acc, 4), lr=sched.get_last_lr()[0],
                       gnorm=round(float(gn), 3), s_per_step=round(dt, 2),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            eta_h = (total_steps - step) * dt / 3600
            print(json.dumps(rec) + f"  eta {eta_h:.1f}h")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            loss_acc, tokens_acc = 0.0, 0

        if step % args.eval_every == 0 or step == total_steps:
            s, hyps, et = evaluate()
            improved = s["wer"] < best_wer
            if improved:
                best_wer = s["wer"]; save("best")
            rec = dict(step=step, eval_wer=round(s["wer"], 4), eval_cer=round(s["cer"], 4), best=improved, eval_s=round(et))
            print("EVAL", json.dumps(rec))
            for r, h in list(zip(eval_recs, hyps))[:3]:
                print(f"   REF {r['raw'][:90]}\n   HYP {h[:90]}")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            d = save("last")
            torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step, best_wer=best_wer),
                       d / "trainer_state.pt")
            t_last, last_log_step = time.time(), step  # don't count eval time as step time
    print(f"done. best WER {best_wer:.4f} -> {run_dir / 'best'}")


if __name__ == "__main__":
    main()

"""QLoRA fine-tuning of the Qwen2.5-Omni-7B Thinker on Algerian Darija speech.

Stage 1 task (this script): Darija audio -> Darija text (listening / transcription). It teaches the
audio encoder and the LLM Darija phonetics and vocabulary; it is the standard first step for
adapting a speech LLM to a new language. LoRA (r=16) on the LLM (q,k,v,o,gate,up,down) and on the
audio encoder (q,k,v,out,fc1,fc2). Loss only on answer tokens (+ <|im_end|>).

Eval: greedy WER/CER on a fixed subset of Casablanca-Algeria validation (human labels, never
trained on), same normalisation as the Whisper baseline.

  python -m omni.train_omni --run omni7b_asr_v1
  python -m omni.train_omni --run omni7b_asr_v1 --resume
  python -m omni.train_omni --run smoke --smoke 4
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
from asr.eval_asr import score, select_eval  # noqa: E402
from common.audio import N_SAMPLES  # noqa: E402
from common.paths import RUNS  # noqa: E402
from common.text_norm import normalize_display  # noqa: E402
from omni.model import LORA_TARGETS, build_prompt, load_processor, load_thinker, transcribe  # noqa: E402


class OmniASRDataset(Dataset):
    def __init__(self, recs, processor, train=True):
        self.recs, self.proc, self.train, self.store = recs, processor, train, None
        self.prompt = build_prompt(processor)
        self.im_end = "<|im_end|>"

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        if self.store is None:
            self.store = AudioStore()
        r = self.recs[i]
        a = self.store.get(r)[:N_SAMPLES]
        if self.train:
            a = a * (10 ** (random.uniform(-6, 6) / 20))
            peak = np.abs(a).max()
            if peak > 1.0: a = a / peak
        enc = self.proc(text=[self.prompt], audio=[a], return_tensors="pt", sampling_rate=16000)
        ans = self.proc.tokenizer(normalize_display(r["raw"]) + self.im_end, add_special_tokens=False,
                                  return_tensors="pt").input_ids
        p = enc["input_ids"].shape[1]
        input_ids = torch.cat([enc["input_ids"], ans], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, p:] = ans
        return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                    input_features=enc["input_features"], feature_attention_mask=enc["feature_attention_mask"])


def collate_one(batch):
    assert len(batch) == 1, "batch size 1 + gradient accumulation (variable lengths, 8 GB GPU)"
    return batch[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2500, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=8, help="clips per optimizer step")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--vram-fraction", type=float, default=0.9)
    ap.add_argument("--mix", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    from common import winrun
    winrun.apply(vram_fraction=args.vram_fraction)
    from peft import LoraConfig, get_peft_model

    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_f = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = "cuda"

    processor = load_processor()
    mix = json.loads(args.mix) if args.mix else DEFAULT_MIX
    train_recs, report = select_train(mix)
    print(f"train set: {len(train_recs)} clips, {sum(v['hours'] for v in report.values()):.1f} h  {report}")
    eval_recs = select_eval("casablanca_val", 8 if args.smoke else args.eval_limit)
    store = AudioStore()
    eval_wavs = [store.get(r)[:N_SAMPLES] for r in eval_recs]

    t0 = time.time()
    last = run_dir / "last"
    resuming = args.resume and (last / "adapter_config.json").exists()
    model = load_thinker(adapter=str(last) if resuming else None, trainable=True)
    print(f"loaded thinker in {time.time()-t0:.0f}s; GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not resuming:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
                         target_modules=LORA_TARGETS)
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    base = model.get_base_model()
    head = base.lm_head  # HeadSwitch

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    total = args.smoke or args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    step, best_wer = 0, float("inf")
    if resuming and (last / "trainer_state.pt").exists():
        st = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best_wer = st["step"], st["best_wer"]
        print(f"resumed at step {step}, best WER {best_wer:.4f}")

    ds = OmniASRDataset(train_recs, processor)
    g = torch.Generator(); g.manual_seed(step)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_one, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0, generator=g,
                    prefetch_factor=4 if args.workers else None)

    def evaluate():
        model.eval(); base.config.use_cache = True
        t = time.time()
        hyps = transcribe(model, processor, eval_wavs, batch_size=4)
        s = score([r["raw"] for r in eval_recs], hyps)
        model.train(); base.config.use_cache = False
        return s, hyps, time.time() - t

    def save(name):
        d = run_dir / name
        model.save_pretrained(str(d))
        return d

    if args.smoke:
        s, hyps, et = evaluate()
        print(f"EVAL before training (n={len(eval_recs)}): WER {s['wer']:.4f} CER {s['cer']:.4f} ({et:.0f}s)")

    model.train()
    it = iter(dl)
    t_last, last_step, loss_sum, n_tok, n_clips = time.time(), step, 0.0, 0, 0
    print(f"start: step {step}/{total}, {args.accum} clips/step, lr {args.lr}")
    while step < total:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            labels = b.pop("labels")
            b["input_features"] = b["input_features"].to(torch.bfloat16)
            head.passthrough = True  # forward returns hidden states; project only answer positions
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(**b, use_cache=False).logits  # (1, T, 3584)
            head.passthrough = False
            tgt = labels[:, 1:]
            mask = tgt != -100
            h = hidden[:, :-1][mask]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head.head(h)
            loss = F.cross_entropy(logits.float(), tgt[mask])
            (loss / args.accum).backward()
            loss_sum += loss.item() * int(mask.sum()); n_tok += int(mask.sum()); n_clips += 1
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1

        if step % 10 == 0 or step == 1 or args.smoke:
            dt = (time.time() - t_last) / max(1, step - last_step)
            t_last, last_step = time.time(), step
            rec = dict(step=step, loss=round(loss_sum / max(1, n_tok), 4), lr=sched.get_last_lr()[0],
                       gnorm=round(float(gn), 3), s_per_step=round(dt, 1), clips=n_clips,
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            print(json.dumps(rec) + f"  eta {(total-step)*dt/3600:.1f}h")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            loss_sum, n_tok, n_clips = 0.0, 0, 0

        if step % args.eval_every == 0 or step == total:
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
            t_last, last_step = time.time(), step
    print(f"done. best WER {best_wer:.4f} -> {run_dir / 'best'}")


if __name__ == "__main__":
    main()

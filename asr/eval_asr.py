"""WER / CER of a Whisper model (optionally + LoRA adapter) on Casablanca-Algeria.

Both reference and hypothesis go through normalize_for_wer (diacritics, punctuation, hamza/alef,
ta-marbuta folded) so the score reflects recognition rather than spelling convention.

  python -m asr.eval_asr --model openai/whisper-large-v3-turbo --split casablanca_val --limit 300
  python -m asr.eval_asr --model openai/whisper-large-v3-turbo --adapter <run>/best --split casablanca_test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.data import AudioStore, load_index  # noqa: E402
from common.audio import N_SAMPLES, LogMel  # noqa: E402
from common.paths import HF_CACHE  # noqa: E402
from common.text_norm import normalize_for_wer  # noqa: E402


def select_eval(split: str, limit: int, seed: int = 0):
    recs = [r for r in load_index(split) if normalize_for_wer(r["raw"])]
    if limit and limit < len(recs):
        rng = np.random.default_rng(seed)  # fixed subset so numbers are comparable across runs
        recs = [recs[i] for i in sorted(rng.choice(len(recs), limit, replace=False))]
    return recs


@torch.no_grad()
def transcribe(model, processor, recs, batch_size=16, device="cuda", max_new_tokens=220):
    logmel = LogMel(processor.feature_extractor, device)
    store = AudioStore()
    order = sorted(range(len(recs)), key=lambda i: recs[i]["n"])
    hyps = [None] * len(recs)
    dtype = next(model.parameters()).dtype
    for s in range(0, len(order), batch_size):
        ids = order[s: s + batch_size]
        wav = np.zeros((len(ids), N_SAMPLES), np.float32)
        for j, i in enumerate(ids):
            a = store.get(recs[i])[:N_SAMPLES]
            wav[j, :len(a)] = a
        feats = logmel(torch.from_numpy(wav).to(device)).to(dtype)
        out = model.generate(input_features=feats, language="ar", task="transcribe",
                             max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        texts = processor.batch_decode(out, skip_special_tokens=True)
        for j, i in enumerate(ids):
            hyps[i] = texts[j].strip()
    return hyps


def score(refs_raw, hyps_raw):
    refs = [normalize_for_wer(r) for r in refs_raw]
    hyps = [normalize_for_wer(h) for h in hyps_raw]
    return dict(wer=jiwer.wer(refs, hyps), cer=jiwer.cer(refs, hyps))


def load_model(model_id, adapter=None, device="cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(adapter or model_id, cache_dir=str(HF_CACHE)) \
        if adapter and (Path(adapter) / "tokenizer_config.json").exists() \
        else WhisperProcessor.from_pretrained(model_id, cache_dir=str(HF_CACHE))
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, cache_dir=str(HF_CACHE))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.generation_config.forced_decoder_ids = None
    return model.to(device).eval(), processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="casablanca_val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="jsonl of predictions")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from common import winrun
    winrun.apply(pin_pcores=False)

    recs = select_eval(args.split, args.limit)
    model, processor = load_model(args.model, args.adapter)
    t0 = time.time()
    hyps = transcribe(model, processor, recs, args.batch_size)
    s = score([r["raw"] for r in recs], hyps)
    print(f"{args.model} adapter={args.adapter} {args.split} n={len(recs)}: "
          f"WER={s['wer']:.4f} CER={s['cer']:.4f} ({time.time()-t0:.0f}s)")
    for r, h in list(zip(recs, hyps))[:8]:
        print(f"  REF: {normalize_for_wer(r['raw'])}\n  HYP: {normalize_for_wer(h)}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r, h in zip(recs, hyps):
                f.write(json.dumps(dict(id=r["id"], ref=r["raw"], hyp=h), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

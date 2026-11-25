"""WER / CER of the Qwen2.5-Omni-7B Thinker (optionally + LoRA adapter) on Casablanca-Algeria.
Same clips (seeded subset) and same normalisation as asr/eval_asr.py, so numbers are comparable
with the Whisper baseline.

  python -m omni.eval_omni --split casablanca_val --limit 300
  python -m omni.eval_omni --adapter C:/Users/seddi/darija_data/runs/omni7b_asr_v1/best --split casablanca_test
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.data import AudioStore  # noqa: E402
from asr.eval_asr import score, select_eval  # noqa: E402
from common.audio import N_SAMPLES  # noqa: E402
from common.text_norm import normalize_for_wer  # noqa: E402
from omni.model import load_processor, load_thinker, transcribe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="casablanca_val")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from common import winrun
    winrun.apply(vram_fraction=0.9, pin_pcores=False)

    recs = select_eval(args.split, args.limit)
    store = AudioStore()
    wavs = [store.get(r)[:N_SAMPLES] for r in recs]
    proc = load_processor()
    model = load_thinker(adapter=args.adapter).eval()
    t0 = time.time()
    hyps = transcribe(model, proc, wavs, batch_size=args.batch_size)
    s = score([r["raw"] for r in recs], hyps)
    print(f"Qwen2.5-Omni-7B adapter={args.adapter} {args.split} n={len(recs)}: "
          f"WER={s['wer']:.4f} CER={s['cer']:.4f} ({time.time()-t0:.0f}s)")
    for r, h in list(zip(recs, hyps))[:8]:
        print(f"  REF: {normalize_for_wer(r['raw'])}\n  HYP: {normalize_for_wer(h)}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r, h in zip(recs, hyps):
                f.write(json.dumps(dict(id=r["id"], ref=r["raw"], hyp=h), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

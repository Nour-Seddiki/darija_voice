"""Which open speech codec should the Darija/French/Arabic voice model speak in?

For each language bucket (dz = Algerian Darija, fr = French, ar = MSA/Classical Arabic) take N
prepared clips, push them through each codec (encode -> discrete tokens -> decode) and measure
what survives:
  * intelligibility: Whisper-large-v3-turbo CER on the reconstruction vs on the original
    (dCER = how many extra character errors the codec causes; lower is better)
  * consistency: CER between Whisper's transcript of the original and of the reconstruction
  * speaker similarity: cosine of WavLM x-vectors (microsoft/wavlm-base-plus-sv), original vs recon
  * cost: discrete tokens per second (what the voice LLM must generate)
WAV pairs are saved for listening.

Note: most Darija sources are natively 16 kHz, so clips are upsampled 16 -> 24 kHz before the
24 kHz codecs. That is the real condition the voice model will be trained under.

  python -m codec.bench_codecs --per-source 15
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.data import AudioStore, load_index  # noqa: E402
from common.audio import N_SAMPLES, LogMel  # noqa: E402
from common.paths import DATA_DIR, HF_CACHE  # noqa: E402
from common.text_norm import normalize_for_wer  # noqa: E402

BUCKETS = {"dz": ["kahwa", "casablanca_val"], "fr": ["siwis_fr", "mls_fr"], "ar": ["cv_ar", "clartts_ar"]}
WHISPER_LANG = {"dz": "ar", "ar": "ar", "fr": "fr"}
DEV = "cuda"
MODELS = DATA_DIR / "models"  # filled by: python -m common.robust_download model <repo>


def local(repo: str) -> str:
    d = MODELS / repo.split("/")[-1]
    return str(d) if (d / "config.json").exists() else repo


class Mimi:
    sr = 24000

    def __init__(self, n_q):
        from transformers import MimiModel
        self.m = MimiModel.from_pretrained(local("kyutai/mimi"), cache_dir=str(HF_CACHE)).to(DEV).eval()
        self.n_q, self.name = n_q, f"mimi_{n_q}q"
        self.tok_per_s = 12.5 * n_q

    @torch.no_grad()
    def roundtrip(self, x24):
        t = torch.from_numpy(x24)[None, None].to(DEV)
        codes = self.m.encode(t, num_quantizers=self.n_q).audio_codes
        y = self.m.decode(codes).audio_values[0, 0, : len(x24)]
        return y.float().cpu().numpy(), int(codes.numel())


class Snac:
    sr, name = 24000, "snac_24k"

    def __init__(self):
        from snac import SNAC
        self.m = SNAC.from_pretrained(local("hubertsiuzdak/snac_24khz")).to(DEV).eval()
        self.tok_per_s = None  # measured

    @torch.no_grad()
    def roundtrip(self, x24):
        t = torch.from_numpy(x24)[None, None].to(DEV)
        codes = self.m.encode(t)
        y = self.m.decode(codes)[0, 0, : len(x24)]
        return y.float().cpu().numpy(), int(sum(c.numel() for c in codes))


def pick_clips(per_source, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for lang, srcs in BUCKETS.items():
        for s in srcs:
            try:
                recs = [r for r in load_index(s) if 3.0 <= r["dur"] <= 15.0 and normalize_for_wer(r["raw"])]
            except Exception:
                recs = []
            if not recs:
                print(f"  (no prepared clips for {s} yet - skipped)")
                continue
            for i in rng.choice(len(recs), min(per_source, len(recs)), replace=False):
                r = dict(recs[i]); r["bucket"] = lang
                out.append(r)
    return out


class Whisper:
    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        mid = "openai/whisper-large-v3-turbo"
        self.p = WhisperProcessor.from_pretrained(mid, cache_dir=str(HF_CACHE))
        self.m = WhisperForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, cache_dir=str(HF_CACHE)).to(DEV).eval()
        self.m.generation_config.forced_decoder_ids = None
        self.mel = LogMel(self.p.feature_extractor, DEV)

    @torch.no_grad()
    def __call__(self, x16, lang):
        w = np.zeros((1, N_SAMPLES), np.float32); w[0, : min(len(x16), N_SAMPLES)] = x16[:N_SAMPLES]
        f = self.mel(torch.from_numpy(w).to(DEV)).to(torch.bfloat16)
        out = self.m.generate(input_features=f, language=lang, task="transcribe", max_new_tokens=200, num_beams=1)
        return self.p.batch_decode(out, skip_special_tokens=True)[0].strip()


class SpeakerSim:
    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        mid = local("microsoft/wavlm-base-plus-sv")
        self.fe = AutoFeatureExtractor.from_pretrained(mid, cache_dir=str(HF_CACHE))
        self.m = WavLMForXVector.from_pretrained(mid, cache_dir=str(HF_CACHE)).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, a16, b16):
        inp = self.fe([a16, b16], sampling_rate=16000, return_tensors="pt", padding=True).to(DEV)
        e = torch.nn.functional.normalize(self.m(**inp).embeddings, dim=-1)
        return float((e[0] * e[1]).sum())


def cer(ref, hyp):
    r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
    return jiwer.cer(r, h) if r else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=15)
    ap.add_argument("--codecs", nargs="+", default=["mimi_8q", "mimi_32q", "snac_24k"])
    ap.add_argument("--save-wavs", type=int, default=2, help="examples per language to save")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = DATA_DIR / "codec_bench"; out_dir.mkdir(parents=True, exist_ok=True)

    clips = pick_clips(args.per_source)
    print(f"{len(clips)} clips: " + ", ".join(f"{b}={sum(c['bucket']==b for c in clips)}" for b in BUCKETS))
    store = AudioStore()
    asr, spk = Whisper(), SpeakerSim()

    # transcribe originals once
    for c in clips:
        c["x16"] = store.get(c)
        c["hyp_orig"] = asr(c["x16"], WHISPER_LANG[c["bucket"]])

    codecs = []
    for name in args.codecs:
        codecs.append(Mimi(int(name.split("_")[1][:-1])) if name.startswith("mimi") else Snac())

    rows, saved = [], {b: 0 for b in BUCKETS}
    for c in clips:
        x24 = soxr.resample(c["x16"], 16000, 24000, quality="HQ").astype(np.float32)
        save = saved[c["bucket"]] < args.save_wavs
        if save:
            k = saved[c["bucket"]]; saved[c["bucket"]] += 1
            sf.write(out_dir / f"{c['bucket']}_{k}_original.wav", x24, 24000)
            (out_dir / f"{c['bucket']}_{k}_text.txt").write_text(c["raw"], encoding="utf-8")
        for cd in codecs:
            t0 = time.time()
            y24, n_tok = cd.roundtrip(x24)
            dt = time.time() - t0
            y16 = soxr.resample(y24, 24000, 16000, quality="HQ").astype(np.float32)
            hyp = asr(y16, WHISPER_LANG[c["bucket"]])
            rows.append(dict(codec=cd.name, bucket=c["bucket"], src=c["src"], dur=c["dur"],
                             cer_orig=cer(c["raw"], c["hyp_orig"]), cer_recon=cer(c["raw"], hyp),
                             consistency=cer(c["hyp_orig"], hyp), spk_sim=spk(c["x16"], y16),
                             tok_per_s=n_tok / c["dur"], rtf=dt / c["dur"]))
            if save:
                sf.write(out_dir / f"{c['bucket']}_{k}_{cd.name}.wav", y24, 24000)

    # summary
    summary = {}
    print(f"\n{'codec':10s} {'lang':4s} {'n':>3s} {'CER orig':>8s} {'CER recon':>9s} {'dCER':>6s} {'consist':>7s} {'spkSim':>6s} {'tok/s':>6s}")
    for cd in codecs:
        for b in BUCKETS:
            rs = [r for r in rows if r["codec"] == cd.name and r["bucket"] == b]
            if not rs: continue
            m = lambda k: float(np.nanmean([r[k] for r in rs]))
            s = dict(n=len(rs), cer_orig=m("cer_orig"), cer_recon=m("cer_recon"), dcer=m("cer_recon") - m("cer_orig"),
                     consistency=m("consistency"), spk_sim=m("spk_sim"), tok_per_s=m("tok_per_s"), rtf=m("rtf"))
            summary[f"{cd.name}/{b}"] = s
            print(f"{cd.name:10s} {b:4s} {s['n']:3d} {s['cer_orig']:8.3f} {s['cer_recon']:9.3f} {s['dcer']:+6.3f} "
                  f"{s['consistency']:7.3f} {s['spk_sim']:6.3f} {s['tok_per_s']:6.1f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"\nWAV examples + summary in {out_dir}")


if __name__ == "__main__":
    main()

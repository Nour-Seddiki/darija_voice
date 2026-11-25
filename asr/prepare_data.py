"""Download + prepare the speech sources for the three languages: Algerian Darija (dz),
French (fr) and Modern Standard / Classical Arabic (ar).

For each source: download one parquet shard at a time, decode every clip to 16 kHz mono,
write it as raw int16 to PREPARED/<source>/<shard>.bin, write one jsonl line per clip
(offsets, duration, raw + normalised text), then delete the parquet. Resumable: finished
shards are recorded in PREPARED/<source>/done.txt.

Source audit (2026-11-25, see README): quantumpuls3/Algerian-STT-* are mostly NOT Algerian
(Egyptian "Darija" rows, Moroccan "Adiren", Saudi "SADA22") and embed the Casablanca test set,
so only their 182 CAFE rows are used. UBC-NLP/Casablanca (Algeria) is CC-BY-NC-ND and
human-labelled -> evaluation only, never training.
French/MSA audit: MASC (MohamedRashad/MASC-Arabic) is NOT pure MSA (contains Egyptian etc.), so
the MSA bucket uses Common Voice 17 Arabic (CC0, many read-speech speakers) + ClArTTS (CC-BY,
one studio speaker, fully diacritised). French = MLS French (CC-BY, audiobooks, ~31 h per shard,
capped by max_files) + SIWIS (CC-BY, one studio speaker, punctuated).

Usage:
  python -m asr.prepare_data --sources casablanca_val casablanca_test
  python -m asr.prepare_data --sources kahwa yasmine rawi cafe loubna
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.audio import SR, decode_to_16k, to_int16  # noqa: E402
from common.robust_download import download_file  # noqa: E402
from common.paths import DOWNLOADS, PREPARED  # noqa: E402
from common.text_norm import normalize_target  # noqa: E402

SOURCES = {
    # --- Algerian Darija (dz) ---
    "kahwa":   dict(lang="dz", repo="oddadmix/arabic-audio-collection-algerian-kahwa-postcast", text="transcript_text", spk="original_video_id"),
    "rawi":    dict(lang="dz", repo="oddadmix/arabic-audio-collection-algerian-rawi", text="transcript_text", spk="original_video_id"),
    "loubna":  dict(lang="dz", repo="oddadmix/arabic-audio-collection-algerian-loubna-stories", text="transcript_text", spk="original_video_id"),
    "yasmine": dict(lang="dz", repo="yasminekaced/algerian_tts", text="text", spk="video_id", merge=True),
    "cafe":    dict(lang="dz", repo="quantumpuls3/Algerian-STT-Super-Dataset-V2", text="transcription", only_source="CAFE"),
    "casablanca_val":  dict(lang="dz", repo="UBC-NLP/Casablanca", text="transcription", prefix="Algeria/validation-", spk="gender"),
    "casablanca_test": dict(lang="dz", repo="UBC-NLP/Casablanca", text="transcription", prefix="Algeria/test-", spk="gender"),
    # --- French (fr) ---
    "mls_fr":   dict(lang="fr", repo="facebook/multilingual_librispeech", text="transcript", spk="speaker_id",
                     prefix="french/train-", max_files=4),   # ~31 h per shard -> ~125 h
    "siwis_fr": dict(lang="fr", repo="Rcarvalo/siwis-fr-24k", text="transcript", spk="speaker_id"),
    # --- Modern Standard / Classical Arabic (ar) ---
    "cv_ar":      dict(lang="ar", repo="fixie-ai/common_voice_17_0", text="sentence", spk="client_id", prefix="ar/train-"),
    "clartts_ar": dict(lang="ar", repo="MBZUAI/ClArTTS", text="text", prefix="data/train-",
                       raw_array_sr="sampling_rate"),  # audio stored as a float64 list + per-row sample rate
}


def merge_contiguous(rows, max_span=28.0, max_gap=0.5):
    """yasmine = YouTube *rolling* captions: caption k is displayed from start_k for dur_k, but the
    next caption already starts at start_{k+1} < start_k + dur_k (~2.5 s overlap). So a raw clip
    contains speech that belongs to the NEXT caption's text -> misaligned pairs.
    Fix: caption k's speech is audio[start_k : start_{k+1}]; stitch those slices into runs of
    consecutive captions (<= max_span s). A run ends at the start of the caption after it, so no
    run contains speech whose text is missing. Silences > max_gap s between captions split runs."""
    rows = sorted(rows, key=lambda r: (r["spk"], float(r["start"] or 0)))
    pieces = []  # (spk, piece_audio, text, id, gap_after_seconds or None at video end)
    for k, r in enumerate(rows):
        st, a = float(r["start"] or 0), r["audio"]
        du = len(a) / SR
        nxt = rows[k + 1] if k + 1 < len(rows) and rows[k + 1]["spk"] == r["spk"] else None
        if nxt is None:
            pieces.append((r["spk"], a, r["raw"], r["id"], None)); continue
        step = float(nxt["start"] or 0) - st
        if 0 < step <= du:          # overlapping / abutting captions: keep only our own slice
            pieces.append((r["spk"], a[: int(round(step * SR))], r["raw"], r["id"], 0.0))
        elif step > du:             # silence between captions: whole clip, then a gap
            pieces.append((r["spk"], a, r["raw"], r["id"], step - du))
        else:                       # out-of-order timestamps: keep as a standalone clip
            pieces.append((r["spk"], a, r["raw"], r["id"], None))
    out, cur = [], None
    for spk, a, txt, rid, gap_after in pieces:
        if cur is not None and (sum(len(x) for x in cur["audio"]) + len(a)) / SR <= max_span:
            cur["audio"].append(a); cur["raw"].append(txt); cur["ids"].append(rid)
        else:
            if cur: out.append(cur)
            cur = dict(spk=spk, audio=[a], raw=[txt], ids=[rid])
        if gap_after is None or gap_after > max_gap:  # close the run
            out.append(cur); cur = None
    if cur: out.append(cur)
    return [dict(id=c["ids"][0] + f"+{len(c['ids'])-1}", spk=c["spk"], audio=np.concatenate(c["audio"]),
                 raw=" ".join(c["raw"])) for c in out]


class Writer:
    """One .bin + one index part per parquet file, written as .tmp and renamed when the file is
    complete, so a crash never leaves duplicated or half-written clips behind."""
    def __init__(self, d: Path, stem: str, lang: str):
        self.final_bin, self.final_idx = d / f"{stem}.bin", d / f"{stem}.jsonl"
        self.tmp_bin, self.tmp_idx = d / f"{stem}.bin.tmp", d / f"{stem}.jsonl.tmp"
        self.fh = open(self.tmp_bin, "wb"); self.idx = open(self.tmp_idx, "w", encoding="utf-8")
        self.stem, self.off, self.lang = stem, 0, lang

    def add(self, rec_id, spk, audio_f32, raw_text):
        a = to_int16(audio_f32)
        self.fh.write(a.tobytes())
        self.idx.write(json.dumps(dict(id=rec_id, spk=spk, lang=self.lang, bin=self.final_bin.name, off=self.off, n=len(a),
                                       dur=round(len(a) / SR, 3), raw=raw_text,
                                       text=normalize_target(raw_text)), ensure_ascii=False) + "\n")
        self.off += len(a)

    def commit(self):
        for f in (self.fh, self.idx):
            f.flush(); os.fsync(f.fileno()); f.close()
        os.replace(self.tmp_bin, self.final_bin); os.replace(self.tmp_idx, self.final_idx)


def _audio_bytes(a):
    """HF Audio columns are either a {bytes, path} struct or plain binary (quantumpuls3)."""
    return a if isinstance(a, (bytes, bytearray)) else a["bytes"]


def _audio_path(a):
    return None if isinstance(a, (bytes, bytearray)) else a.get("path")


def _safe_decode(a, raw_sr=None):
    try:
        if raw_sr is not None:  # ClArTTS: raw sample array (float64 list) at raw_sr Hz
            import soxr
            x = np.asarray(a, dtype=np.float32)
            if raw_sr != SR:
                x = soxr.resample(x, raw_sr, SR, quality="HQ").astype(np.float32)
        else:
            x = decode_to_16k(_audio_bytes(a))
        return x if len(x) >= SR // 10 and np.isfinite(x).all() else None
    except Exception as e:  # a corrupt clip must not kill a multi-hour prep run
        print(f"  ! decode failed ({None if raw_sr else _audio_path(a)}): {e}", flush=True)
        return None


def decode_table(tbl, cfg, pool):
    cols = tbl.column_names
    audios = tbl.column("audio").to_pylist()
    if cfg.get("raw_array_sr"):
        srs = [int(s) for s in tbl.column(cfg["raw_array_sr"]).to_pylist()]
        wavs = list(pool.map(_safe_decode, audios, srs))
        audios = [None] * len(audios)  # no path/bytes struct for ids
    else:
        wavs = list(pool.map(_safe_decode, audios))
    texts = tbl.column(cfg["text"]).to_pylist()
    spk = tbl.column(cfg["spk"]).to_pylist() if cfg.get("spk") in cols else [None] * len(texts)
    if "file" in cols and cfg.get("raw_array_sr"):
        ids = [str(x) for x in tbl.column("file").to_pylist()]
    else:
        ids = [((_audio_path(a) if a is not None else None) or str(i)) for i, a in enumerate(audios)]
    starts = tbl.column("start").to_pylist() if "start" in cols else [0] * len(texts)
    return [dict(id=i, spk=s, audio=w, raw=t or "", start=st)
            for i, s, w, t, st in zip(ids, spk, wavs, texts, starts) if w is not None]


def process_source(name, args):
    cfg = SOURCES[name]
    out = PREPARED / name
    out.mkdir(parents=True, exist_ok=True)
    done_f = out / "done.txt"
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                   if f.endswith(".parquet") and f.startswith(cfg.get("prefix", "")))
    limit = args.max_files or cfg.get("max_files", 0)  # --max-files overrides the source's own cap
    if limit: files = files[:limit]
    print(f"[{name}] {len(files)} parquet files, {len(done)} already done", flush=True)
    fs = HfFileSystem()
    with ThreadPoolExecutor(args.workers) as pool:
        for f in files:
            if f in done: continue
            t0 = time.time(); n_in = n_out = 0; secs = 0.0
            w = Writer(out, Path(f).stem, cfg["lang"])
            if cfg.get("only_source"):
                # Read just the row groups containing the wanted source, without downloading 21 GB.
                with fs.open(f"datasets/{cfg['repo']}/{f}", "rb", block_size=1 << 22) as fh:
                    pf = pq.ParquetFile(fh)
                    for g in range(pf.num_row_groups):
                        src = pf.read_row_group(g, columns=["source"]).column("source").to_pylist()
                        if cfg["only_source"] not in src: continue
                        tbl = pf.read_row_group(g)
                        keep = [i for i, s in enumerate(src) if s == cfg["only_source"]]
                        tbl = tbl.take(keep)
                        rows = decode_table(tbl, cfg, pool); n_in += len(rows)
                        for r in rows:
                            w.add(r["id"], r["spk"], r["audio"], r["raw"]); n_out += 1; secs += len(r["audio"]) / SR
            else:
                local = download_file(cfg["repo"], f, DOWNLOADS / name)  # watchdog + retries
                pf = pq.ParquetFile(local)
                rows_all = []
                for g in range(pf.num_row_groups):
                    rows = decode_table(pf.read_row_group(g), cfg, pool); n_in += len(rows)
                    if cfg.get("merge"):
                        rows_all.extend(rows)
                    else:
                        for r in rows:
                            w.add(r["id"], r["spk"], r["audio"], r["raw"]); n_out += 1; secs += len(r["audio"]) / SR
                if cfg.get("merge"):
                    for r in merge_contiguous(rows_all):
                        w.add(r["id"], r["spk"], r["audio"], r["raw"]); n_out += 1; secs += len(r["audio"]) / SR
                del pf
                if not args.keep_parquet:
                    Path(local).unlink(missing_ok=True)
            w.commit()
            with open(done_f, "a") as d: d.write(f + "\n")
            print(f"[{name}] {f}: {n_in} rows -> {n_out} clips, {secs/3600:.2f} h, {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, choices=list(SOURCES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=0, help="debug: only the first N parquet files")
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()
    for s in args.sources:
        process_source(s, args)


if __name__ == "__main__":
    main()

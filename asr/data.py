"""Read the prepared shards (see prepare_data.py)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PREPARED  # noqa: E402


def load_index(source: str) -> list[dict]:
    from asr.prepare_data import SOURCES  # lang per source; older Darija shards have no "lang" field
    lang = SOURCES[source]["lang"]
    d = PREPARED / source
    recs = []
    for f in sorted(d.glob("*.jsonl")):  # committed shards only (.jsonl.tmp is ignored)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                r["src"] = source
                r.setdefault("lang", lang)
                r["bin_path"] = str(d / r["bin"])
                recs.append(r)
    return recs


# Hours cap per source (None = all). Loubna is one speaker and 236 h; uncapped she would be
# >50% of the data and the model would over-fit her voice and story-telling style.
DEFAULT_MIX = {"kahwa": None, "rawi": None, "yasmine": None, "cafe": None, "loubna": 60}


def select_train(mix: dict, seed: int = 0, max_dur: float = 30.0):
    """Filtered, capped, shuffled training records + a per-source report.
    Drops clips outside 1..max_dur s, empty transcripts, and transcript/audio mismatches
    (letters per second outside 2..30: truncated or hallucinated captions)."""
    import random
    from common.text_norm import arabic_char_count
    rng = random.Random(seed)
    recs, report = [], {}
    for src, cap_h in mix.items():
        rs = load_index(src)
        ok = [r for r in rs if 1.0 <= r["dur"] <= max_dur and r["text"]
              and 2.0 <= arabic_char_count(r["text"]) / r["dur"] <= 30.0]
        rng.shuffle(ok)
        if cap_h is not None:
            kept, tot = [], 0.0
            for r in ok:
                if tot >= cap_h * 3600: break
                kept.append(r); tot += r["dur"]
            ok = kept
        report[src] = dict(rows=len(rs), kept=len(ok), hours=round(sum(r["dur"] for r in ok) / 3600, 1))
        recs.extend(ok)
    rng.shuffle(recs)
    return recs, report


class AudioStore:
    """Lazily memory-maps each .bin; safe to use inside DataLoader workers."""
    def __init__(self):
        self._maps = {}

    def get(self, rec) -> np.ndarray:
        m = self._maps.get(rec["bin_path"])
        if m is None:
            m = self._maps[rec["bin_path"]] = np.memmap(rec["bin_path"], dtype=np.int16, mode="r")
        return m[rec["off"]: rec["off"] + rec["n"]].astype(np.float32) / 32768.0

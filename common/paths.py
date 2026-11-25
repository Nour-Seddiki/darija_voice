"""Where big files live. Kept OUT of the OneDrive-synced project folder on purpose:
the prepared audio is tens of GB and must not be uploaded by OneDrive."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DARIJA_DATA", r"C:\Users\seddi\darija_data"))
PREPARED = DATA_DIR / "prepared"      # 16 kHz int16 shards + jsonl indexes, one folder per source
DOWNLOADS = DATA_DIR / "downloads"    # temporary parquet downloads (deleted after processing)
RUNS = DATA_DIR / "runs"              # checkpoints, logs
HF_CACHE = DATA_DIR / "hf_cache"      # model weights

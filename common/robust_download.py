"""Resumable, verified Hugging Face downloads for an unstable connection.

Why not huggingface_hub: on this laptop (2026-11-25) the HF servers reset the connection every few
minutes (WinError 10054). huggingface_hub then starts a NEW .incomplete file on the next attempt and
deletes the old one, so a 5 GB shard never finishes (3.9 GB were lost that way). Xet transfers also
raised CAS errors or sat at 0 bytes on disk.

How this works:
  * each file is split into byte ranges (64 MB); each range downloads into its own part file and
    resumes from that part's current size with an HTTP Range request -> a reset costs only the
    bytes in flight, also across process restarts
  * read timeout 60 s, unlimited retries with backoff per range (a stall becomes a retry)
  * parts are concatenated and the result is checked against the size and, for LFS files, the
    SHA-256 published by the Hub before it is renamed into place

  python -m common.robust_download model Qwen/Qwen2.5-Omni-7B     # -> DATA_DIR/models/Qwen2.5-Omni-7B
"""
import fnmatch
import hashlib
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from huggingface_hub import HfApi

SEG = 64 * 1024 * 1024
_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=60.0)


def _url(repo, filename, repo_type, revision="main"):
    pre = "" if repo_type == "model" else f"{repo_type}s/"
    return f"https://huggingface.co/{pre}{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"


def _meta(repo, filename, repo_type):
    info = HfApi().get_paths_info(repo, [filename], repo_type=repo_type, expand=True)[0]
    sha = info.lfs.sha256 if getattr(info, "lfs", None) else None
    return int(info.size), sha


def _fetch(client, url, start, end, part: Path, progress, lock):
    need = end - start + 1
    tries = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have == need:
            return
        if have > need:  # should never happen; restart this range cleanly
            part.unlink(); have = 0
        try:
            with client.stream("GET", url, headers={"Range": f"bytes={start + have}-{end}"}) as r:
                if r.status_code == 200 and not (start + have == 0):
                    raise RuntimeError("server ignored the Range header")
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(part, "ab") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        if have + len(chunk) > need:
                            chunk = chunk[: need - have]
                        f.write(chunk); have += len(chunk)
                        with lock: progress[0] += len(chunk)
                        if have >= need:
                            break
            tries = 0
        except Exception as e:
            tries += 1
            if tries % 5 == 1:
                print(f"    range {start}-{end}: {type(e).__name__}: {str(e)[:120]} (retry {tries})", flush=True)
            time.sleep(min(60, 2 * tries))


def download_file(repo, filename, local_dir, repo_type="dataset", workers=6) -> str:
    """Download repo/filename to local_dir/filename (resumable, verified). Returns the path."""
    local_dir = Path(local_dir)
    dest = local_dir / filename
    size, sha = _meta(repo, filename, repo_type)
    if dest.exists() and dest.stat().st_size == size:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = dest.parent / f".{dest.name}.parts"
    parts_dir.mkdir(exist_ok=True)
    ranges = [(s, min(s + SEG, size) - 1) for s in range(0, size, SEG)] or [(0, -1)]
    url = _url(repo, filename, repo_type)
    progress, lock = [sum((parts_dir / f"{i:05d}").stat().st_size for i in range(len(ranges))
                          if (parts_dir / f"{i:05d}").exists())], threading.Lock()
    done_evt = threading.Event()

    def report():
        t0, b0 = time.time(), progress[0]
        while not done_evt.wait(30):
            rate = (progress[0] - b0) / max(1e-6, time.time() - t0) / 1e6
            print(f"  {filename}: {progress[0]/1e9:.2f}/{size/1e9:.2f} GB  {rate:.1f} MB/s", flush=True)

    if size > 0:
        threading.Thread(target=report, daemon=True).start()
        with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client, ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda ir: _fetch(client, url, ir[1][0], ir[1][1], parts_dir / f"{ir[0]:05d}", progress, lock),
                          enumerate(ranges)))
        done_evt.set()
    # assemble + verify
    tmp = dest.parent / f".{dest.name}.assembling"
    h = hashlib.sha256()
    with open(tmp, "wb") as out:
        for i in range(len(ranges) if size > 0 else 0):
            with open(parts_dir / f"{i:05d}", "rb") as f:
                while True:
                    b = f.read(8 << 20)
                    if not b: break
                    out.write(b); h.update(b)
    got = tmp.stat().st_size
    if got != size or (sha and h.hexdigest() != sha):
        tmp.unlink()
        for p in parts_dir.iterdir(): p.unlink()
        raise RuntimeError(f"{filename}: verification failed (size {got} vs {size}, sha ok={not sha or h.hexdigest()==sha}); parts cleared")
    os.replace(tmp, dest)
    for p in parts_dir.iterdir(): p.unlink()
    parts_dir.rmdir()
    return str(dest)


def download_repo(repo, local_dir, allow, repo_type="model") -> str:
    files = [f for f in HfApi().list_repo_files(repo, repo_type=repo_type)
             if any(fnmatch.fnmatch(f, p) for p in allow)]
    files.sort(key=lambda f: f.endswith(".safetensors"))  # small config files first
    for f in files:
        t = time.time()
        p = download_file(repo, f, local_dir, repo_type=repo_type)
        print(f"OK {f} ({os.path.getsize(p)/1e6:.1f} MB, {time.time()-t:.0f}s)", flush=True)
    return str(local_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.paths import DATA_DIR
    kind, repo = sys.argv[1], sys.argv[2]
    allow = sys.argv[3].split(",") if len(sys.argv) > 3 else ["*.json", "*.safetensors", "*.txt", "*.jinja", "spk_dict.pt"]
    out = DATA_DIR / "models" / repo.split("/")[-1]
    download_repo(repo, out, allow=allow, repo_type=kind)
    print("DONE", out, flush=True)

#!/usr/bin/env python3
"""
Unified weight download script for GenArtist.
Uses config.MODEL_ZOO_DIR; supports resume and file-existence checks.
Run from project root: python tools/download_weights.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import Optional

# Ensure project root is on path for config import
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MODEL_ZOO_DIR


# ---------- HuggingFace (snapshot_download) ----------
HF_MODELS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "stable-diffusion-xl-base-1.0"),
    ("stabilityai/stable-diffusion-2-1-base", "stable-diffusion-2-1-base"),
    ("stabilityai/stable-diffusion-2", "stable-diffusion-2"),
    ("facebook/sam-vit-base", "sam-vit-base"),
]

# ---------- Direct / non-HF (requests stream + optional zip) ----------
# (url, dest_path_relative_to_MODEL_ZOO_DIR, optional_expected_size_bytes)
DIRECT_DOWNLOADS = [
    (
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        Path("GroundingDINO/weights/groundingdino_swint_ogc.pth"),
        None,
    ),
    (
        "https://huggingface.co/osunlp/InstructPix2Pix-MagicBrush/resolve/main/MagicBrush-epoch-000168.ckpt",
        Path("MagicBrush/MagicBrush-epoch-000168.ckpt"),
        None,
    ),
]

# LaMa: zip from HF, unzip to MODEL_ZOO_DIR/LaMa/big-lama
LAMA_ZIP_URL = "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
LAMA_EXTRACT_DIR = Path("LaMa")  # under MODEL_ZOO_DIR; zip extracts to big-lama/ inside it


def _log(msg: str) -> None:
    print(f"[download_weights] {msg}")


def download_hf(repo_id: str, local_subdir: str) -> bool:
    """Download HuggingFace model with snapshot_download; skip if dir already present and non-empty."""
    import huggingface_hub

    dest = MODEL_ZOO_DIR / local_subdir
    if dest.exists():
        # Consider "present" if there is at least one file (snapshot_download creates multiple)
        if any(dest.iterdir()):
            _log(f"SKIP (already present): {repo_id} -> {dest}")
            return True
    _log(f"Downloading HF: {repo_id} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        _log(f"OK: {repo_id}")
        return True
    except Exception as e:
        _log(f"FAIL: {repo_id} -> {e}")
        return False


def _stream_download(
    url: str,
    dest_path: Path,
    expected_size: Optional[int] = None,
) -> bool:
    """Stream download with resume (Range) and file-existence check."""
    import requests

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    try:
        r = requests.get(url, stream=True, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        _log(f"FAIL request: {url} -> {e}")
        return False

    total = r.headers.get("Content-Length")
    if total is not None:
        total = int(total)
        if existing_size == total:
            _log(f"SKIP (already complete): {dest_path.name}")
            return True
        # If we got Range, total is the remaining size
        if r.status_code == 206:
            total = existing_size + total
    else:
        total = None

    mode = "ab" if r.status_code == 206 and existing_size > 0 else "wb"
    written = existing_size
    chunk_size = 1024 * 1024  # 1 MiB

    try:
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total is not None:
                    pct = min(100, round(100 * written / total, 1))
                    print(f"\r  {dest_path.name}: {written / (1024**2):.1f} MiB / {total / (1024**2):.1f} MiB ({pct}%)", end="", flush=True)
        if total is not None:
            print()
        if expected_size is not None and written != expected_size:
            _log(f"WARN: size mismatch {written} vs expected {expected_size}")
        _log(f"OK: {dest_path}")
        return True
    except Exception as e:
        _log(f"FAIL write: {dest_path} -> {e}")
        return False


def download_direct(url: str, relative_path: Path, expected_size: Optional[int]) -> bool:
    """Download one file to MODEL_ZOO_DIR / relative_path with resume support."""
    import requests

    dest = MODEL_ZOO_DIR / relative_path
    if dest.exists():
        local_size = dest.stat().st_size
        if expected_size is not None and local_size == expected_size:
            _log(f"SKIP (already complete): {relative_path}")
            return True
        try:
            head = requests.head(url, allow_redirects=True, timeout=15)
            if head.status_code == 200:
                remote = head.headers.get("Content-Length")
                if remote and int(remote) == local_size:
                    _log(f"SKIP (already complete): {relative_path}")
                    return True
        except Exception:
            pass
    _log(f"Downloading: {url} -> {dest}")
    return _stream_download(url, dest, expected_size)


def download_lama() -> bool:
    """Download LaMa big-lama zip and extract to MODEL_ZOO_DIR/LaMa/big-lama."""
    dest_dir = MODEL_ZOO_DIR / LAMA_EXTRACT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Check if already extracted
    big_lama = dest_dir / "big-lama"
    if big_lama.is_dir() and (big_lama / "config.yaml").exists():
        _log(f"SKIP (already present): LaMa big-lama -> {big_lama}")
        return True

    zip_path = dest_dir / "big-lama.zip"
    _log(f"Downloading LaMa: {LAMA_ZIP_URL} -> {zip_path}")
    if not _stream_download(LAMA_ZIP_URL, zip_path, expected_size=None):
        return False
    _log("Extracting big-lama.zip ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        # Zip may contain top-level "big-lama/" or flat files; ensure we have config.yaml
        big_lama = dest_dir / "big-lama"
        if not big_lama.is_dir():
            # Zip might have extracted as root; check for config.yaml in dest_dir
            if (dest_dir / "config.yaml").exists():
                big_lama = dest_dir
        if big_lama.is_dir() and (big_lama / "config.yaml").exists():
            _log(f"OK: LaMa big-lama -> {big_lama}")
        else:
            _log("OK: LaMa extracted (verify config.yaml inside)")
        return True
    except Exception as e:
        _log(f"FAIL extract: {e}")
        return False
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except OSError:
                pass


def main() -> None:
    _log(f"MODEL_ZOO_DIR = {MODEL_ZOO_DIR}")
    MODEL_ZOO_DIR.mkdir(parents=True, exist_ok=True)

    failed = []

    # 1) HuggingFace
    for repo_id, local_subdir in HF_MODELS:
        if not download_hf(repo_id, local_subdir):
            failed.append(repo_id)

    # 2) Direct files
    for url, rel_path, expected_size in DIRECT_DOWNLOADS:
        if not download_direct(url, rel_path, expected_size):
            failed.append(url)

    # 3) LaMa zip + extract
    if not download_lama():
        failed.append("LaMa big-lama")

    if failed:
        _log(f"Some downloads failed: {failed}")
        sys.exit(1)
    _log("All weights are in place.")


if __name__ == "__main__":
    main()

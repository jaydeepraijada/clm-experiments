"""Shared paths and helpers for running the pipeline on one machine (the DGX Spark).

All data lives under DATA_DIR (env var, default ~/clm-data). Layout:
  qa/shard_XX/chunk_NNNN_{q,a}.npy, _meta.parquet   published Q&A vectors
  neg/shard_00/chunk_NNNN_{n,ncnt}.npy              hard negatives
  heads/                                            the authors' published heads
  agent/pairs, agent/vec, agent/vec2                post-training data
  runs/<run>/{latest,best}.pt, log.jsonl            training runs
"""
import os
import tempfile

DATA_DIR = os.path.abspath(os.path.expanduser(os.environ.get("DATA_DIR", "~/clm-data")))
QWEN = "Qwen/Qwen3-8B"
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"
CLM_REPO = "https://github.com/Contrastive-LM/CLM.git"
CLM_COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"


def data(*parts):
    return os.path.join(DATA_DIR, *parts)


def qwen_path():
    """Local directory of Qwen3-8B at the pinned revision; uses the HF cache and downloads only if it is missing."""
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
    try:
        return snapshot_download(QWEN, revision=QWEN_REV, allow_patterns=pats, local_files_only=True)
    except Exception:
        print(f"{QWEN}@{QWEN_REV[:8]} not in the HF cache, downloading (~16 GB)", flush=True)
        return snapshot_download(QWEN, revision=QWEN_REV, allow_patterns=pats)


def atomic_save(write, path):
    """write(tmp_path) then rename, so an interrupted run never leaves a half-written output that looks complete."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=os.path.splitext(path)[1])
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_npy(path, arr):
    import numpy as np
    atomic_save(lambda t: np.save(t, arr), path)


def clm_repo():
    """Checkout of the authors' CLM repo at the pinned commit (under DATA_DIR/src), cloned on first use."""
    import subprocess
    dst = data("src", "CLM")
    if not os.path.exists(os.path.join(dst, ".git")):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["git", "clone", CLM_REPO, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", CLM_COMMIT], check=True)
    return dst

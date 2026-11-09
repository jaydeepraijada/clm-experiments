"""Download the published Q&A vectors, hard negatives and reference heads into DATA_DIR. Safe to rerun:
files already present at the right size are skipped, and each file lands with an atomic rename.

    python pipeline/stage_data.py --dry-run                         # plan: what is missing, and whether it fits
    python pipeline/stage_data.py --only shard_00/chunk_0028        # a slice (prefix match on the repo path)
    python pipeline/stage_data.py                                   # everything (~1.3 TB)
"""
import argparse
import os
import shutil
import time

from common import data

QA = "Contrastive-LM/CLM-v0.1-Pretrain-Nemotron"
QA_REV = "05ca7d05c043350326c5e86c589b9542962747b4"
NEG = "jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings"
HEADS = "jackyk02/qwen3_8b_head"
HEAD_FILES = ("qwen3_8b_pretrained_head.pt", "qwen3_8b_midtrained_head.pt", "qwen3_8b_posttrained_head.pt")


def listing(api, repo, rev, pat):
    from huggingface_hub.hf_api import RepoFile
    return {x.path: x.size for x in api.list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True)
            if isinstance(x, RepoFile) and x.path.endswith(pat)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", default="qa,neg,heads", help="comma list of qa, neg, heads")
    ap.add_argument("--only", default="", help="comma list of path prefixes, e.g. shard_00/chunk_0000,shard_05/chunk_0083")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the disk check, download nothing")
    ap.add_argument("--force", action="store_true", help="download even if the missing bytes exceed free disk")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    kinds = set(args.kinds.split(","))
    only = [p for p in args.only.split(",") if p]
    api = HfApi()
    jobs = []  # (repo, rev, kind, path, size)
    if "qa" in kinds:
        jobs += [(QA, QA_REV, "qa", f, s) for f, s in sorted(listing(api, QA, QA_REV, (".npy", ".parquet")).items())]
    if "neg" in kinds:
        jobs += [(NEG, None, "neg", f, s) for f, s in sorted(listing(api, NEG, None, (".npy", ".parquet", "metadata.json")).items())]
    if only:
        jobs = [j for j in jobs if any(j[3].startswith(p) for p in only)]

    def have(j):
        dst = data(j[2], j[3])
        return os.path.exists(dst) and os.path.getsize(dst) == j[4]

    todo = [j for j in jobs if not have(j)]
    need = sum(j[4] for j in todo)
    os.makedirs(data(), exist_ok=True)
    free = shutil.disk_usage(data()).free
    for k in ("qa", "neg"):
        sel = [j for j in jobs if j[2] == k]
        if sel:
            print(f"{k}: {len(sel)} files, {sum(j[4] for j in sel)/1e9:.1f} GB selected, "
                  f"{sum(1 for j in todo if j[2] == k)} missing", flush=True)
    print(f"to download {need/1e9:.1f} GB | free on {data()}: {free/1e9:.1f} GB", flush=True)
    if args.dry_run:
        return
    if need > free - 20e9 and not args.force:
        raise SystemExit("not enough disk (keeping a 20 GB margin); free space or pass --force")

    staging = data(".staging")
    t0, got = time.time(), 0
    for i, (repo, rev, kind, f, size) in enumerate(todo):
        dst = data(kind, f)
        p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=os.path.join(staging, kind))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(p, dst)  # same filesystem: no second copy
        got += size
        mins = (time.time() - t0) / 60
        print(f"{i+1}/{len(todo)} {kind}/{f} | {got/1e9:.1f}/{need/1e9:.1f} GB, {got/1e6/max(mins*60,1):.0f} MB/s, {mins:.1f} min", flush=True)
    shutil.rmtree(staging, ignore_errors=True)

    if "heads" in kinds and not only:
        os.makedirs(data("heads"), exist_ok=True)
        for f in HEAD_FILES:
            if not os.path.exists(data("heads", f)):
                shutil.copyfile(hf_hub_download(HEADS, f), data("heads", f))
        print("heads in", data("heads"), flush=True)
    print(f"DONE {got/1e9:.1f} GB new in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

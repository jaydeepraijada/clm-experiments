"""Copy the published Q&A vectors and hard negatives into a Modal Volume. Safe to rerun."""
import os, shutil, tempfile, time
import modal
TORCH = ["torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0", "huggingface_hub==0.26.2", "hf_transfer==0.1.8"]
image = modal.Image.debian_slim(python_version="3.11").pip_install(*TORCH).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
QA = "Contrastive-LM/CLM-v0.1-Pretrain-Nemotron"
QA_REV = "05ca7d05c043350326c5e86c589b9542962747b4"
NEG = "jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings"
HEADS = "jackyk02/qwen3_8b_head"
app = modal.App("clm-stage-data")

@app.function(image=image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=3600, retries=modal.Retries(max_retries=6, initial_delay=30.0), max_containers=16)
def fetch(job):
    from huggingface_hub import hf_hub_download
    repo, rev, kind, files, sizes = job
    done = 0
    for f, size in zip(files, sizes):
        dst = f"/data/{kind}/{f}"
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=tmp)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(p, dst + ".part"); os.replace(dst + ".part", dst)
        done += size
    vol.commit()
    return done

@app.function(image=image, volumes={"/data": vol}, timeout=86400)
def orchestrate():
    from huggingface_hub import HfApi, hf_hub_download
    import shutil
    api = HfApi(); jobs = []
    def listing(repo, rev, pat):
        out = {}
        for x in api.list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True):
            if hasattr(x, "size") and x.path.endswith(pat): out[x.path] = x.size
        return out
    qa = listing(QA, QA_REV, (".npy", ".parquet")); neg = listing(NEG, None, (".npy", ".parquet", "metadata.json"))
    print("qa files", len(qa), round(sum(qa.values())/1e12, 3), "TB | neg files", len(neg), round(sum(neg.values())/1e12, 3), "TB", flush=True)
    by = {}
    for f, s in qa.items():
        by.setdefault(f.split("/")[-1].split("_")[1] + f.split("/")[0], []).append((f, s))
    for group in by.values():
        jobs.append((QA, QA_REV, "qa", [g[0] for g in group], [g[1] for g in group]))
    for f, s in neg.items():
        jobs.append((NEG, None, "neg", [f], [s]))
    t = time.time(); total = 0
    for i, n in enumerate(fetch.map(jobs, order_outputs=False, return_exceptions=True)):
        if isinstance(n, Exception): print("job failed", repr(n)[:200], flush=True); continue
        total += n
        if i % 25 == 0: print(f"{i}/{len(jobs)} jobs, {total/1e9:.0f} GB new, {(time.time()-t)/60:.1f} min", flush=True)
    vol.reload()
    os.makedirs("/data/heads", exist_ok=True)
    for f in ("qwen3_8b_pretrained_head.pt", "qwen3_8b_midtrained_head.pt", "qwen3_8b_posttrained_head.pt"):
        shutil.copyfile(hf_hub_download(HEADS, f), "/data/heads/" + f)
    vol.commit()
    print(f"DONE {total/1e9:.0f} GB new in {(time.time()-t)/60:.1f} min", flush=True)

@app.local_entrypoint()
def main():
    orchestrate.remote()

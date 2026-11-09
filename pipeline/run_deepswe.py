"""Fine-tune a head on DeepSWE with the repo's own script, then score the 38 held-out tasks.

    modal run --detach run_deepswe.py --tag authors --init authors    # control: should come near 31/38
    modal run --detach run_deepswe.py --tag ours --init /data/runs/post/best.pt

The README's evaluation dataset returns 401, so evaluation reads the public copy, tarsur385/deepswe-prm-embeddings-8k.
The head is trained the way its own metadata says: batch 512, seed 1234, passing rollouts of the training tasks.
"""
import json
import os
import subprocess
import sys

import modal

COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .pip_install("torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0", "huggingface_hub>=0.34,<1.0", "requests==2.32.3", "transformers==4.57.6")
         .run_commands(f"git clone https://github.com/Contrastive-LM/CLM.git /clm && git -C /clm checkout {COMMIT}"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-deepswe")


def sh(args, tail=2500):
    r = subprocess.run(args, capture_output=True, text=True, cwd="/clm")
    print(r.stdout[-tail:], r.stderr[-800:], flush=True)
    return r.stdout


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=8, memory=64000, timeout=7200)
def run(tag, init, seed=1234):
    from huggingface_hub import hf_hub_download, snapshot_download
    h = snapshot_download("Contrastive-LM/deepswe-clm-heads-8k", local_dir="/tmp/heads")
    path = hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt") if init == "authors" else init
    out = f"/data/deepswe/{tag}"
    os.makedirs(out, exist_ok=True)
    # Their head records training_pairs 22576 and 66 tasks with successes: passing rollouts of the 75 training
    # tasks, taken from the evaluation rollouts. The README's default dataset (405,919 steps) does not reproduce it.
    sys.path.insert(0, "/clm/preprocessing")
    import hf_embeddings, torch
    src = hf_embeddings.download("tarsur385/deepswe-prm-embeddings-8k", "/tmp/deepswe_eval")
    meta = json.load(open(src + "/metadata.json"))["samples"]
    heldout = set(json.load(open(h + "/heldout_tasks.json")))
    keep = [i for i, m in enumerate(meta) if m["task_id"] not in heldout and float(m.get("reward") or 0) > 0.5]
    print("training pairs", len(keep), "tasks", len({meta[i]["task_id"] for i in keep}), "(their head: 22576, 66)", flush=True)
    os.makedirs("/tmp/deepswe_train", exist_ok=True)
    idx = torch.tensor(keep)
    torch.save(torch.load(src + "/state_embeddings.pt")[idx], "/tmp/deepswe_train/state_embeddings.pt")
    torch.save(torch.load(src + "/action_embeddings.pt")[idx], "/tmp/deepswe_train/action_embeddings.pt")
    json.dump({"num_samples": len(keep), "hidden_size": 4096, "samples": [meta[i] for i in keep]}, open("/tmp/deepswe_train/metadata.json", "w"))
    sh([sys.executable, "train/finetune.py", "--task", "clm", "--init-ckpt", path, "--out-dir", out, "--emb-dir", "/tmp/deepswe_train",
        "--holdout-tasks", h + "/heldout_tasks.json", "--batch", "512", "--seed", str(seed), "--epochs", "20", "--patience", "5"], 900)
    vol.commit()
    res = sh([sys.executable, "evaluation/bon_eval.py", "--hf-dataset", "tarsur385/deepswe-prm-embeddings-8k",
              "--checkpoint", out + "/best_head.pt", "--tasks-file", h + "/heldout_tasks.json", "--n", "4", "--window", "12"])
    print("RESULT", [l for l in res.splitlines() if "[bon]" in l and ("%" in l or "random" in l)], flush=True)
    vol.commit()


@app.local_entrypoint()
def main(tag: str = "authors", init: str = "authors", seed: int = 1234):
    run.remote(tag, init, seed)

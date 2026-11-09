"""Embed the agent (state, action) pairs with frozen Qwen3-8B, using the repo's token recipe:
last-token pooling, L2 normalized, float16. States keep their most recent 2047 tokens, actions their
first 2047. We cap at 2048 tokens, the serving default, to keep this affordable.

    modal run --detach embed_agent.py
"""
import glob
import json
import os
import time

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm==0.11.0", "transformers>=4.56,<5", "pyarrow==18.1.0", "huggingface_hub")
         .env({"HF_HOME": "/data/hf"}))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-embed-agent")
REV = "b968826d9c46dd6066d109eabc6255188de91218"
CAP = 2047


@app.cls(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=48000, timeout=7200, max_containers=8,
         scaledown_window=120)
class Embedder:
    @modal.enter()
    def load(self):
        from transformers import AutoTokenizer
        from vllm import LLM
        vol.reload()
        self.tok = AutoTokenizer.from_pretrained("/data/qwen3-8b")
        self.llm = LLM(model="/data/qwen3-8b", runner="pooling", max_model_len=2048,
                       gpu_memory_utilization=0.85, enforce_eager=False)

    def embed(self, ids):
        import numpy as np
        from vllm.inputs import TokensPrompt
        out = []
        for i in range(0, len(ids), 4096):
            res = self.llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids[i:i + 4096]], use_tqdm=False)
            out.append(np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in res]))
        v = np.concatenate(out)
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)).astype(np.float16)

    def state_ids(self, raw):
        msgs = json.loads(raw)
        kept, total = [], 0
        for m in reversed(msgs):  # the recipe keeps the tail, so older text never reaches the encoder
            kept.append(m); total += len(m["content"])
            if total > 16000:
                break
        kept.reverse()
        try:
            ids = self.tok.apply_chat_template(kept, tokenize=True, add_generation_prompt=False)
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
        except Exception:
            ids = self.tok("\n\n".join(m["content"] for m in kept), add_special_tokens=False)["input_ids"]
        return list(ids)[-CAP:]

    @modal.method()
    def run(self, path):
        import numpy as np, pyarrow as pa, pyarrow.parquet as pq
        name = os.path.basename(path)[:-8]
        out = f"/data/agent/vec/{name}"
        vol.reload()
        if os.path.exists(out + "_meta.parquet"):
            return name, 0, 0.0
        t0 = time.time()
        tab = pq.read_table(path)
        st = [self.state_ids(s) for s in tab.column("state").to_pylist()]
        ac = [self.tok(a or " ", add_special_tokens=False)["input_ids"][:CAP] or [220] for a in tab.column("action").to_pylist()]
        t1 = time.time()
        q, a = self.embed(st), self.embed(ac)
        os.makedirs("/data/agent/vec", exist_ok=True)
        np.save(out + "_q.npy", q); np.save(out + "_a.npy", a)
        pq.write_table(tab.select(["config", "traj", "step", "split"]), out + "_meta.parquet")
        vol.commit()
        toks = sum(map(len, st)) + sum(map(len, ac))
        print(f"{name}: {len(st)} pairs, {toks/1e6:.1f}M tokens, tok {t1-t0:.0f}s, embed {time.time()-t1:.0f}s", flush=True)
        return name, len(st), time.time() - t0


@app.function(image=image, volumes={"/data": vol}, timeout=14000)
def orchestrate():
    from huggingface_hub import snapshot_download
    vol.reload()
    snapshot_download("Qwen/Qwen3-8B", revision=REV, local_dir="/data/qwen3-8b", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    vol.commit()
    files = sorted(glob.glob("/data/agent/pairs/*.parquet"))
    print(len(files), "configs", flush=True)
    total = 0
    for r in Embedder().run.map(files, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:300], flush=True)
        else:
            total += r[1]; print(r, "total", total, flush=True)
    print("DONE", total, flush=True)


@app.local_entrypoint()
def main():
    orchestrate.remote()

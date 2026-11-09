"""Second reconstruction of the agent actions: only the decision itself (the tool call, or the text after
"Action:"), without the agent's narration. The authors' exact conversion is not public; the game's options are
short decisions, so this tests whether long narrated actions caused our gap on the game. States are reused."""
import glob, json, os, re, time
import modal
image = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.11.0", "transformers>=4.56,<5", "pyarrow==18.1.0", "huggingface_hub"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-embed-actions")
CALL = re.compile(r"^[\w\-.]+\(\{.*\}\)$|^[\w\-.]+\(null\)$", re.S)

def decision(a):
    lines = (a or "").split("\n")
    for i, l in enumerate(lines):
        if re.match(r"^[\w\-.]+\((\{|null)", l):
            return "\n".join(lines[i:]).strip()
    if "Action:" in a:
        return a.rsplit("Action:", 1)[1].strip()
    return a.strip() or " "

@app.cls(image=image, volumes={"/data": vol}, gpu="H100", cpu=12, memory=64000, timeout=3000, max_containers=10, scaledown_window=60)
class E:
    @modal.enter()
    def load(self):
        from transformers import AutoTokenizer
        from vllm import LLM
        vol.reload()
        self.tok = AutoTokenizer.from_pretrained("/data/qwen3-8b")
        self.llm = LLM(model="/data/qwen3-8b", runner="pooling", max_model_len=2048, gpu_memory_utilization=0.85)

    @modal.method()
    def run(self, path):
        import numpy as np, pyarrow.parquet as pq
        from vllm.inputs import TokensPrompt
        name = os.path.basename(path)[:-8]; out = f"/data/agent/vec2/{name}_a.npy"
        vol.reload()
        if os.path.exists(out):
            return name, 0
        acts = [decision(a) for a in pq.read_table(path, columns=["action"]).column("action").to_pylist()]
        ids = [self.tok(a, add_special_tokens=False)["input_ids"][:512] or [220] for a in acts]
        vs = []
        for i in range(0, len(ids), 4096):
            res = self.llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids[i:i + 4096]], use_tqdm=False)
            vs.append(np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in res]))
        v = np.concatenate(vs); v = (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)).astype(np.float16)
        os.makedirs("/data/agent/vec2", exist_ok=True); np.save(out, v); vol.commit()
        print(name, len(acts), "tokens", sum(map(len, ids)), "sample:", acts[0][:80].replace("\n", " "), flush=True)
        return name, len(acts)

@app.function(image=image, volumes={"/data": vol}, timeout=3200)
def orchestrate():
    vol.reload(); n = 0
    for r in E().run.map(sorted(glob.glob("/data/agent/pairs/*.parquet")), order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception): print("FAILED", repr(r)[:200], flush=True)
        else: n += r[1]; print(r, "total", n, flush=True)
    print("DONE", n, flush=True)

@app.local_entrypoint()
def main(): orchestrate.remote()

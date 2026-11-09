"""Play the real T-Rex harness briefly with many heads, to see which training stage changes decision quality."""
import json, os, subprocess, sys, time
import modal
COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone https://github.com/Contrastive-LM/CLM.git /clm && git -C /clm checkout {COMMIT}",
                       "pip install vllm==0.11.0 'transformers>=4.56,<5' -e /clm huggingface_hub httpx==0.28.1"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-trex-sweep")
SEED = 0

def requests_models():
    import hashlib, requests
    return requests.get("http://127.0.0.1:8700/health", timeout=5).text[:160]

def wait(url, secs=600):
    import requests
    t = time.time()
    while time.time() - t < secs:
        try:
            if requests.get(url, timeout=3).status_code == 200: return
        except Exception: pass
        time.sleep(2)
    raise TimeoutError(url)

@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=48000, timeout=2400)
def sweep(heads):
    log = open("/tmp/e.log", "w")
    e = subprocess.Popen("vllm serve /data/qwen3-8b --served-model-name qwen3-8b --runner pooling --enable-prefix-caching --max-model-len 2048 --port 8090 --gpu-memory-utilization 0.6", shell=True, stdout=log, stderr=log)
    wait("http://127.0.0.1:8090/v1/models", 900)
    out = {}
    for name, path in heads:
        subprocess.run("pkill -f clm-serve; sleep 2", shell=True)  # never let an old head answer for a new one
        s = subprocess.Popen(["clm-serve", "--no-ui", "--host", "127.0.0.1", "--ckpt", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait("http://127.0.0.1:8700/health", 200)
        print("SERVING", name, requests_models(), flush=True)
        for mode, extra in (("shield", []), ("noshield", ["--no-shield"])):
            f = f"/tmp/{name}_{mode}.json"
            r = subprocess.run([sys.executable, "/clm/examples/t_rex/run.py", "--model", "clm", "--seeds", "2", "--duration", "30", "--seed", str(SEED), "--out", f] + extra, capture_output=True, text=True, timeout=300)
            m = json.load(open(f))["summary"] if os.path.exists(f) else {"err": r.stderr[-200:]}
            out[f"{name}/{mode}"] = {k: m.get(k) for k in ("survived", "deaths", "mean_best_score", "mean_agreement_with_planner", "shield_interventions", "err") if k in m}
            print("SWEEP", name, mode, json.dumps(out[f"{name}/{mode}"]), flush=True)
        s.terminate(); s.wait(); time.sleep(1)
    e.terminate()
    return out

@app.local_entrypoint()
def main(heads: str):
    H = [tuple(x.split("=")) for x in heads.split(",")]
    res = {}
    groups = [[h] for h in H]
    for r in sweep.map(groups):
        res.update(r)
    print("RESULT " + json.dumps(res))

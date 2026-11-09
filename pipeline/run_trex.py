"""Play the repo's T-Rex benchmark with a chosen head, recording every run for the side-by-side.

    modal run --detach run_trex.py --tag authors --ckpt authors
    modal run --detach run_trex.py --tag ours --ckpt /data/runs/post/best.pt
    modal run --detach run_trex.py --tag jev --model jev      # needs the Modal secret "typesafe"
"""
import gzip
import json
import os
import subprocess
import sys
import textwrap
import time

import modal

COMMIT = "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone https://github.com/Contrastive-LM/CLM.git /clm && git -C /clm checkout {COMMIT}",
                       "pip install vllm==0.11.0 'transformers>=4.56,<5' -e /clm huggingface_hub httpx==0.28.1"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-trex")

RECORDER = textwrap.dedent("""
    # Runs examples/t_rex/run.py unchanged, with Game.step wrapped to save a light copy of each frame.
    import atexit, gzip, json, os, runpy, sys
    sys.path.insert(0, "/clm/examples/t_rex"); sys.path.insert(0, "/clm/examples")
    from trex.engine import Game
    runs, orig_init, orig_step = [], Game.__init__, Game.step
    def init(self, seed=0, course=None):
        orig_init(self, seed, course); self._rec = {"seed": seed, "frames": []}; runs.append(self._rec)
    def step(self):
        out = orig_step(self); r = self._rec
        if len(r["frames"]) < 7200:
            t = self.trex
            r["frames"].append([round(t.y, 1), int(t.ducking), int(self.crashed), self.score, round(self.speed, 2),
                                [[o.kind.name, round(o.x, 1), o.y, o.width, o.frame] for o in self.obstacles]])
        return out
    Game.__init__, Game.step = init, step
    atexit.register(lambda: gzip.open(os.environ["REC_OUT"], "wt").write(json.dumps(runs)))
    sys.argv = ["run.py"] + sys.argv[1:]
    runpy.run_path("/clm/examples/t_rex/run.py", run_name="__main__")
""")


def play(args, rec, out):
    env = dict(os.environ, REC_OUT=rec)
    r = subprocess.run([sys.executable, "/tmp/rec.py", *args, "--out", out], env=env, capture_output=True, text=True, timeout=1500)
    print(r.stdout[-1800:], r.stderr[-600:], flush=True)


def wait(url, proc, log, seconds=900):
    import requests
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(open(log).read()[-1500:])
        time.sleep(3)
    raise TimeoutError(url)


def body(tag, ckpt, model, seeds):
    vol.reload()
    out = f"/data/trex/{tag}"
    os.makedirs(out, exist_ok=True)
    open("/tmp/rec.py", "w").write(RECORDER)
    procs = []
    if model == "clm":
        log = open("/tmp/enc.log", "w")
        e = subprocess.Popen("vllm serve /data/qwen3-8b --served-model-name qwen3-8b --runner pooling --enable-prefix-caching "
                             "--max-model-len 2048 --port 8090 --gpu-memory-utilization 0.6", shell=True, stdout=log, stderr=log)
        print("encoder up in %.0fs" % wait("http://127.0.0.1:8090/v1/models", e, "/tmp/enc.log"), flush=True)
        from clm.heads import download
        path = download() if ckpt == "authors" else ckpt
        slog = open("/tmp/srv.log", "w")
        s = subprocess.Popen(f"clm-serve --no-ui --host 127.0.0.1 --ckpt {path}", shell=True, stdout=slog, stderr=slog)
        wait("http://127.0.0.1:8700/health", s, "/tmp/srv.log", 300)
        procs += [e, s]
    base = ["--model", model, "--seeds", str(seeds)]
    play(base, f"{out}/shield.json.gz", f"{out}/shield.json")
    vol.commit()
    if model == "clm":
        play(base + ["--no-shield"], f"{out}/noshield.json.gz", f"{out}/noshield.json")
    else:
        play(base + ["--no-shield"], f"{out}/noshield.json.gz", f"{out}/noshield.json")
    vol.commit()
    for p in procs:
        p.terminate()
    res = {}
    for name in ("shield", "noshield"):
        f = f"{out}/{name}.json"
        if os.path.exists(f):
            res[name] = json.load(open(f))["summary"]
    print("RESULT " + json.dumps(res), flush=True)
    return res


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=48000, timeout=3000)
def run_clm(tag, ckpt, seeds):
    return body(tag, ckpt, "clm", seeds)


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=8192, timeout=3000,
              secrets=([modal.Secret.from_name("typesafe")] if os.environ.get("WITH_JEV") else []))
def run_jev(tag, seeds):
    return body(tag, None, "jev", seeds)


@app.local_entrypoint()
def main(tag: str = "authors", ckpt: str = "authors", model: str = "clm", seeds: int = 5):
    print(run_clm.remote(tag, ckpt, seeds) if model == "clm" else run_jev.remote(tag, seeds))

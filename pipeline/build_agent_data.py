"""Turn Agent Data Protocol recordings into (state, action) pairs for post-training.

The authors' conversion and selection are not public, so this is our reconstruction of the
blog's description: "the state contains the agent's current context and the action corresponds
to the decision taken at that step". Choices that are ours:
  - text-only configs; at most 6 random steps per recording, 30,000 pairs per config
  - an action is the agent's message plus its tool calls, rendered as name(json arguments)
  - 3% of recordings held out for validation, chosen by hashing the recording id

    modal run --detach build_agent_data.py
"""
import json
import os
import random
import zlib

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("requests==2.32.3", "pyarrow==18.1.0", "huggingface_hub==0.26.2"))
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-agent-pairs")
REPO = "neulab/adp-v2"
VISION = {"android_in_the_wild", "androidcontrol", "omniact", "screenagent", "llava_plus"}
QUOTA, PER_TRAJ, MAX_CHARS = 30000, 6, 24000


def obs_text(obs):
    if not isinstance(obs, dict):
        return ""
    parts = []
    for r in obs.get("results") or []:
        c = r.get("content")
        if not c:
            c = ((r.get("extra") or {}).get("web") or {}).get("axtree")
        if c:
            parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts)


def action_text(step):
    parts = [step["message"]] if isinstance(step.get("message"), str) and step["message"].strip() else []
    for t in step.get("tool_calls") or []:
        parts.append(f"{t.get('function_name')}({json.dumps(t.get('arguments'), ensure_ascii=False)})")
    return "\n".join(parts).strip()


def lines(url, tries=30):
    """Stream a file's lines; when the connection breaks, reopen with an HTTP Range request at the last byte received.
    The largest ADP files are several GB and a single stream often drops partway."""
    import time
    import requests
    pos, buf, fails = 0, b"", 0
    while True:
        try:
            r = requests.get(url, stream=True, timeout=120, headers={"Range": f"bytes={pos}-"} if pos else {})
            r.raise_for_status()
            if pos and r.status_code != 206:
                raise RuntimeError(f"server ignored the Range request ({r.status_code})")
            for chunk in r.iter_content(1 << 20):
                pos += len(chunk)
                buf += chunk
                *done, buf = buf.split(b"\n")
                yield from done
            r.close()
            if buf:
                yield buf
            return
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            fails += 1
            if fails > tries:
                raise
            print(f"stream broke at {pos/1e9:.2f} GB ({type(e).__name__}), resuming", flush=True)
            time.sleep(min(60, 5 * fails))


def pairs_from(rec, rng):
    """-> [(state messages, action, step index)], at most PER_TRAJ, only for recordings that act repeatedly."""
    hist, out = [], []
    steps = rec.get("steps") or []
    acting = [s for s in steps if s.get("source") == "agent" and action_text(s)]
    if len(acting) < 2:
        return []
    for i, s in enumerate(steps):
        src, msg = s.get("source"), s.get("message")
        if src in ("system", "user"):
            if isinstance(msg, str) and msg:
                hist.append({"role": src, "content": msg[:MAX_CHARS]})
        elif src == "agent":
            a = action_text(s)
            if a:
                if hist:
                    out.append((list(hist), a[:MAX_CHARS], i))
                hist.append({"role": "assistant", "content": a[:MAX_CHARS]})
            o = obs_text(s.get("observation"))
            if o:
                hist.append({"role": "user", "content": o[:MAX_CHARS]})
    rng.shuffle(out)
    return out[:PER_TRAJ]


@app.function(image=image, volumes={"/data": vol}, cpu=2, memory=8192, timeout=7200, max_containers=24,
              retries=modal.Retries(max_retries=3, initial_delay=30.0))
def build(config):
    import pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import HfApi
    rng = random.Random(zlib.crc32(config.encode()))
    files = sorted(x.path for x in HfApi().list_repo_tree(REPO, f"{config}/raw", repo_type="dataset")
                   if x.path.endswith(".jsonl") and "full_std" in x.path)
    rows, recs = [], 0
    for f in files:
        if len(rows) >= QUOTA:
            break
        it = lines(f"https://huggingface.co/datasets/{REPO}/resolve/main/{f}")
        for line in it:
            if len(rows) >= QUOTA:
                break
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            recs += 1
            tid = str(rec.get("trajectory_id") or rec.get("session_id") or recs)
            split = "val" if zlib.crc32(tid.encode()) % 100 < 3 else "train"
            for state, action, step in pairs_from(rec, rng):
                rows.append({"config": config, "traj": tid, "step": step, "split": split,
                             "state": json.dumps(state, ensure_ascii=False), "action": action})
        it.close()
    if rows:
        os.makedirs("/data/agent/pairs", exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), f"/data/agent/pairs/{config}.parquet", compression="zstd")
        vol.commit()
    print(config, "recordings", recs, "pairs", len(rows), flush=True)
    return {"config": config, "recordings": recs, "pairs": len(rows)}


@app.function(image=image, timeout=14000)
def orchestrate(only=""):
    from huggingface_hub import HfApi
    configs = [x.path for x in HfApi().list_repo_tree(REPO, repo_type="dataset") if not x.path.startswith(".") and "." not in x.path]
    configs = [c for c in configs if c not in VISION]
    if only:
        configs = [c for c in configs if c in only.split(",")]
    print(len(configs), "configs", flush=True)
    res = [r for r in build.map(configs, order_outputs=False, return_exceptions=True)]
    ok = [r for r in res if isinstance(r, dict)]
    for r in res:
        if not isinstance(r, dict):
            print("FAILED", repr(r)[:200], flush=True)
    print("TOTAL pairs", sum(r["pairs"] for r in ok), "from", sum(1 for r in ok if r["pairs"]), "configs", flush=True)
    print(json.dumps(sorted(ok, key=lambda r: -r["pairs"])), flush=True)
    return ok


@app.local_entrypoint()
def main(only: str = ""):
    print(orchestrate.remote(only))

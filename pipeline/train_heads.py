"""Train the CLM projection heads from scratch on the published Qwen3-8B vectors.

The authors' repo ships no pre-training or mid-training code. This trainer follows the
configs stored inside their public checkpoints (jackyk02/qwen3_8b_head):

  pre:  batch 16384, lr 0.00653 constant, AdamW, weight decay 0, temperature starts at 0.07
        and is learned (scale capped at 100), same-document pairs masked, head
        4096 -> 1536 -> 1536 (LayerNorm) -> 512, GELU.
  mid:  warm start from pre, shard_00 chunks 0000-0027, batch 16384, lr 0.001 constant,
        up to 10 hard negatives per pair (weight 1.0) added to the state->action direction,
        early stop on hard-negative validation loss.

Held out, as in their notes: shard_00/0028-0029 (hard negatives), shard_05/0083-0084 (general).

    modal run --detach train_heads.py --stage pre
    modal run --detach train_heads.py --stage mid --init /data/runs/pre/best.pt
"""
import glob
import json
import math
import os
import queue
import threading
import time

import modal

TORCH = ["torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0", "huggingface_hub==0.26.2", "hf_transfer==0.1.8"]
image = modal.Image.debian_slim(python_version="3.11").pip_install(*TORCH).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
QA = "Contrastive-LM/CLM-v0.1-Pretrain-Nemotron"
QA_REV = "05ca7d05c043350326c5e86c589b9542962747b4"
NEG = "jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings"
HEADS = "jackyk02/qwen3_8b_head"

app = modal.App("clm-train-heads")
DIM, WIDTH, PROJ = 4096, 1536, 512
HOLD_GEN = {("shard_05", 83), ("shard_05", 84)}
HOLD_HN = {("shard_00", 28), ("shard_00", 29)}


def chunks():
    out = []
    for p in sorted(glob.glob("/data/qa/shard_*/chunk_*_q.npy")):
        out.append((p.split("/")[-2], int(p.split("_")[-2])))
    return out


def stem(c, kind="qa"):
    return f"/data/{kind}/{c[0]}/chunk_{c[1]:04d}"


def read_chunk(c, negs=False):
    import numpy as np, pyarrow.parquet as pq, zlib
    d = {"q": np.load(stem(c) + "_q.npy"), "a": np.load(stem(c) + "_a.npy")}
    ids = pq.read_table(stem(c) + "_meta.parquet", columns=["doc_id"]).column("doc_id").to_pylist()
    d["doc"] = np.fromiter((zlib.crc32(x.encode()) for x in ids), dtype=np.int64, count=len(ids))
    if negs:
        d["n"] = np.load(stem(c, "neg") + "_n.npy")
        d["c"] = np.load(stem(c, "neg") + "_ncnt.npy")
    return d


def make_head():
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(DIM, WIDTH)
            self.hidden = nn.ModuleList([nn.Linear(WIDTH, WIDTH)])
            self.norms = nn.ModuleList([nn.LayerNorm(WIDTH)])
            self.out = nn.Linear(WIDTH, PROJ)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.inp(x))
            x = self.act(self.norms[0](self.hidden[0](x)))
            return self.out(x)

    return Head()


def batches(order, group, batch, negs, seed, stop, repeats=1):
    """One thread reads pools of `group` random chunks, another deals each pool `repeats` times.
    Reusing a pool halves the disk reads per step; every pair is still seen once per pass over a pool."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    pools, q = queue.Queue(maxsize=1), queue.Queue(maxsize=8)

    def put(where, item):
        while not stop.is_set():
            try:
                where.put(item, timeout=1)
                return
            except queue.Full:
                pass

    def reader():
        with ThreadPoolExecutor(group) as ex:
            for i in range(0, len(order), group):
                if stop.is_set():
                    return
                parts = list(ex.map(lambda c: read_chunk(c, negs), order[i:i + group]))
                data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
                del parts
                put(pools, data)
        put(pools, None)

    def dealer():
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            try:
                data = pools.get(timeout=1)
            except queue.Empty:
                continue
            if data is None:
                break
            for _ in range(repeats):
                idx = rng.permutation(len(data["q"]))
                for j in range(0, len(idx) - batch + 1, batch):
                    if stop.is_set():
                        return
                    b = np.sort(idx[j:j + batch])
                    put(q, {k: v[b] for k, v in data.items()})
        put(q, None)

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=dealer, daemon=True).start()
    while not stop.is_set():
        try:
            item = q.get(timeout=5)
        except queue.Empty:
            continue
        if item is None:
            return
        yield item


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=32, memory=256000, timeout=86000,
              retries=modal.Retries(max_retries=4, initial_delay=30.0))
def train(stage="pre", init=None, run=None, batch=16384, lr=None, max_epochs=None, smoke=False, seed=1234, max_steps=None, replay_negs=False, adir="vec"):
    import numpy as np
    import torch
    import torch.nn.functional as F
    dev = "cuda"
    vol.reload()
    run = run or stage
    out = f"/data/runs/{run}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    lr = lr or (2e-3 * math.sqrt(1024 / WIDTH) * math.sqrt(batch / 1024) if stage == "pre" else 1e-3)
    max_epochs = max_epochs or (16 if stage == "pre" else 8)
    sh, ah = make_head().to(dev), make_head().to(dev)
    ls = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev))
    P = list(sh.parameters()) + list(ah.parameters()) + [ls]
    opt = torch.optim.AdamW(P, lr=lr, weight_decay=0.0)
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=True)
        sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
        ls.data.copy_(torch.as_tensor(ck["logit_scale"]).float().to(dev))

    all_chunks = chunks()
    if stage == "pre":
        order0 = [c for c in all_chunks if c not in HOLD_GEN and c not in HOLD_HN]
        group, every, repeats = 24, 1000, 3
    elif stage == "mid":
        order0 = [("shard_00", i) for i in range(28)]
        group, every, repeats = 6, 42, 1
    else:  # post: 60% agent steps, 40% Q&A replay; temperature frozen (their post and mid scales are identical)
        order0 = [c for c in all_chunks if c not in HOLD_GEN and c not in HOLD_HN]
        group, every, repeats = 6, 25, 1
        ls.requires_grad_(False)
        import pyarrow.parquet as pq, zlib
        metas = sorted(glob.glob("/data/agent/vec/*_meta.parquet"))
        aq = np.concatenate([np.load(m.replace("_meta.parquet", "_q.npy")) for m in metas])
        aa = np.concatenate([np.load(m.replace("_meta.parquet", "_a.npy").replace("/agent/vec/", f"/agent/{adir}/")) for m in metas])
        rows = [pq.read_table(m, columns=["traj", "split", "config"]) for m in metas]
        trajs = sum((r.column("traj").to_pylist() for r in rows), [])
        cfgs = sum((r.column("config").to_pylist() for r in rows), [])
        ad = np.fromiter((zlib.crc32((c + t).encode()) for c, t in zip(cfgs, trajs)), dtype=np.int64, count=len(trajs))
        isval = np.array(sum((r.column("split").to_pylist() for r in rows), [])) == "val"
        vq, va, vd = aq[isval], aa[isval], ad[isval]
        aq, aa, ad = aq[~isval], aa[~isval], ad[~isval]
        print(f"agent: {len(aq)} train steps, {len(vq)} val steps from {len(metas)} configs", flush=True)
    print(f"{stage}: {len(order0)} training chunks, lr {lr:.6f}, batch {batch}", flush=True)

    # validation data stays in memory
    gen = [read_chunk(c) for c in sorted(HOLD_GEN)]
    gq, ga, gd = (np.concatenate([g[k] for g in gen]) for k in ("q", "a", "doc"))
    hn = [read_chunk(c, True) for c in sorted(HOLD_HN)]
    hq, ha, hd, hneg, hc = (np.concatenate([h[k] for h in hn]) for k in ("q", "a", "doc", "n", "c"))
    del gen, hn

    def T(x):
        return torch.from_numpy(x).to(dev, non_blocking=True).float()

    @torch.no_grad()
    def eval_gen(xq=None, xa=None, xd=None, prefix="gen"):
        xq, xa, xd = (gq, ga, gd) if xq is None else (xq, xa, xd)
        sh.eval(); ah.eval()
        s = ls.exp().clamp(max=100.0)
        L, hit, tot = [], 0, 0
        perm = np.random.default_rng(0).permutation(len(xq))
        for i in range(0, len(perm) - 1023, 1024):
            b = np.sort(perm[i:i + 1024])
            zq, za = F.normalize(sh(T(xq[b])), dim=-1), F.normalize(ah(T(xa[b])), dim=-1)
            d = torch.from_numpy(xd[b]).to(dev)
            same = d[None] == d[:, None]; same.fill_diagonal_(False)
            lg = (s * zq @ za.t()).masked_fill(same, float("-inf"))
            y = torch.arange(len(b), device=dev)
            L.append(((F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2).item())
            hit += (lg.argmax(1) == y).sum().item(); tot += len(b)
        sh.train(); ah.train()
        return {f"{prefix}_loss": float(np.mean(L)), f"{prefix}_top1": hit / tot}

    @torch.no_grad()
    def eval_hn():
        """One gold answer against its own negatives: the blog's hard-negative top-1 test."""
        sh.eval(); ah.eval()
        s = ls.exp().clamp(max=100.0)
        hit = tot = 0; L = []
        for i in range(0, len(hq), 4096):
            j = min(i + 4096, len(hq))
            c = torch.from_numpy(hc[i:j].astype(np.int64)).to(dev)
            zq, za = F.normalize(sh(T(hq[i:j])), dim=-1), F.normalize(ah(T(ha[i:j])), dim=-1)
            zn = F.normalize(ah(T(hneg[i:j]).view(-1, DIM)), dim=-1).view(j - i, 10, -1)
            pos = (zq * za).sum(-1, keepdim=True)
            neg = torch.einsum("bh,bkh->bk", zq, zn).masked_fill(torch.arange(10, device=dev)[None] >= c[:, None], -9.0)
            v = c > 0
            hit += (pos[v, 0] > neg[v].max(1).values).sum().item(); tot += v.sum().item()
            lg = s * torch.cat([pos, neg], 1)
            lg = lg.masked_fill(torch.cat([torch.zeros_like(pos, dtype=torch.bool), neg <= -9.0], 1), float("-inf"))
            L.append(F.cross_entropy(lg[v], torch.zeros(v.sum(), dtype=torch.long, device=dev)).item())
        sh.train(); ah.train()
        return {"hn_top1": hit / tot, "hn_loss": float(np.mean(L)), "hn_n": tot}

    step, epoch, best, stale = 0, 0, float("inf"), 0
    latest = f"{out}/latest.pt"
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
        ls.data.copy_(ck["logit_scale"].to(dev)); opt.load_state_dict(ck["opt"])
        step, epoch, best, stale = ck["step"], ck["epoch"], ck["best"], ck["stale"]
        print(f"resumed at step {step}, epoch {epoch}", flush=True)

    def blob():
        return {"state_head": sh.state_dict(), "action_head": ah.state_dict(), "logit_scale": ls.detach().cpu(),
                "hidden_size": DIM, "projection_dim": PROJ, "step": step, "epoch": epoch,
                "cfg": {"model": "Qwen/Qwen3-8B", "hidden_size": DIM, "projection_dim": PROJ, "width": WIDTH, "depth": 3,
                        "activation": "gelu", "layernorm": True, "residual": False, "batch": batch, "lr": lr, "stage": stage}}

    log = open(f"{out}/log.jsonl", "a")

    def checkpoint(full):
        nonlocal best, stale
        m = eval_gen()
        if full or stage != "pre":
            m.update(eval_hn())
        if stage == "post":
            m.update(eval_gen(vq, va, vd, "agent"))
        score = {"pre": m["gen_loss"], "mid": m.get("hn_loss"), "post": m.get("agent_loss")}[stage]
        m.update(step=step, epoch=epoch, scale=ls.exp().item(), minutes=(time.time() - t0) / 60)
        if score < best - 1e-4:
            best, stale = score, 0
            torch.save(blob(), f"{out}/best.pt")
            m["best"] = True
        print(json.dumps(m), flush=True)
        log.write(json.dumps(m) + "\n"); log.flush()
        torch.save({**blob(), "opt": opt.state_dict(), "best": best, "stale": stale}, latest)
        vol.commit()

    t0 = time.time()
    if step == 0:
        checkpoint(True)
    limit = {"pre": 9, "mid": 8, "post": 8}[stage]
    while epoch < max_epochs and stale < limit and not (max_steps and step >= max_steps):
        rng = np.random.default_rng(seed + epoch)
        order = [order0[i] for i in rng.permutation(len(order0))]
        stop = threading.Event()
        def post_batches():
            n_qa = int(batch * 0.4)
            rg = np.random.default_rng(seed + 7 + step)
            src = [("shard_00", i) for i in rg.permutation(28).tolist()] if replay_negs else order
            for b in batches(src, group, n_qa, replay_negs, seed + epoch, stop):
                i = np.sort(rg.integers(0, len(aq), batch - n_qa))
                y = {"q": np.concatenate([b["q"], aq[i]]), "a": np.concatenate([b["a"], aa[i]]),
                     "doc": np.concatenate([b["doc"], ad[i]])}
                if replay_negs:  # replay rows come first and keep their hard negatives
                    y["n"], y["c"] = b["n"], b["c"]
                yield y
        for b in (post_batches() if stage == "post" else batches(order, group, batch, stage == "mid", seed + epoch, stop, repeats)):
            q_, a_, d_ = T(b["q"]), T(b["a"]), torch.from_numpy(b["doc"]).to(dev)
            zq, za = F.normalize(sh(q_), dim=-1), F.normalize(ah(a_), dim=-1)
            s = ls.exp().clamp(max=100.0)
            lg = s * zq @ za.t()
            same = d_[None] == d_[:, None]; same.fill_diagonal_(False)
            lg = lg.masked_fill(same, float("-inf"))
            y = torch.arange(len(q_), device=dev)
            back = F.cross_entropy(lg.t(), y)
            if "n" in b:
                c = torch.from_numpy(b["c"].astype(np.int64)).to(dev)
                k = len(c)
                zn = F.normalize(ah(T(b["n"]).view(-1, DIM)), dim=-1).view(k, 10, -1)
                ln = (s * torch.einsum("bh,bkh->bk", zq[:k], zn)).masked_fill(
                    torch.arange(10, device=dev)[None] >= c[:, None], float("-inf"))
                if k < len(q_):
                    ln = torch.cat([ln, torch.full((len(q_) - k, 10), float("-inf"), device=dev)])
                forw = F.cross_entropy(torch.cat([lg, ln], 1), y)
            else:
                forw = F.cross_entropy(lg, y)
            loss = (forw + back) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(P, 1.0)
            opt.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} scale {ls.exp().item():.1f} {(time.time()-t0)/60:.1f} min", flush=True)
            if step % every == 0:
                checkpoint(step % (every * 4) == 0)
            if (smoke and step >= 30) or (max_steps and step >= max_steps) or stale >= limit:
                stop.set(); break
        stop.set()
        if smoke:
            break
        epoch += 1
        checkpoint(True)
    checkpoint(True)
    m = {"finished": True, "step": step, "best": best, "minutes": (time.time() - t0) / 60}
    print(json.dumps(m), flush=True)
    log.write(json.dumps(m) + "\n"); log.close(); vol.commit()
    return m


@app.local_entrypoint()
def main(stage: str = "pre", init: str = "", run: str = "", smoke: bool = False, max_epochs: int = 0, max_steps: int = 0, lr: float = 0.0, replay_negs: bool = False, adir: str = "vec"):
    print(train.remote(stage=stage, init=init or None, run=run or None, smoke=smoke, max_epochs=max_epochs or None, max_steps=max_steps or None, lr=lr or None, replay_negs=replay_negs, adir=adir))

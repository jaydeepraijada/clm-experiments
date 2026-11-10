"""Score any set of heads on the same three held-out tests, with one code path.

    modal run final_eval.py --heads "/data/heads/qwen3_8b_midtrained_head.pt,/data/runs/mid/best.pt"
    modal run final_eval.py --heads "raw,/data/heads/qwen3_8b_pretrained_head.pt,/data/runs/pre/best.pt" --out pre_compare

"raw" scores Qwen3-8B's vectors with plain cosine and no heads. It has no learned scale, so its losses are reported
at the best of a few fixed scales (a best case for the baseline). The agent test runs only when /data/agent/vec has
validation rows, i.e. after post-training data exists.
"""
import glob
import json
import zlib

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy==2.1.3", "pyarrow==18.1.0")
vol = modal.Volume.from_name("clm-data", create_if_missing=True, version=2)
app = modal.App("clm-final-eval")
DIM = 4096


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=96000, timeout=1800)
def run(heads, out_name="final_eval"):
    import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda"
    vol.reload()

    def head():
        class H(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(DIM, 1536); s.hidden = nn.ModuleList([nn.Linear(1536, 1536)]); s.norms = nn.ModuleList([nn.LayerNorm(1536)])
                s.out = nn.Linear(1536, 512); s.act = nn.GELU()

            def forward(s, x):
                x = s.act(s.inp(x)); x = s.act(s.norms[0](s.hidden[0](x))); return s.out(x)
        return H().to(dev).eval()

    T = lambda x: torch.from_numpy(x).to(dev).float()
    def rd(c):
        st = f"/data/qa/{c}"
        ids = pq.read_table(st + "_meta.parquet", columns=["doc_id"]).column("doc_id").to_pylist()
        return np.load(st + "_q.npy"), np.load(st + "_a.npy"), np.fromiter((zlib.crc32(x.encode()) for x in ids), dtype=np.int64, count=len(ids))
    g = [rd(f"shard_05/chunk_{i:04d}") for i in (83, 84)]
    gq, ga, gd = (np.concatenate([x[k] for x in g]) for k in range(3))
    h = [rd(f"shard_00/chunk_{i:04d}") for i in (28, 29)]
    hq, ha = (np.concatenate([x[k] for x in h]) for k in range(2))
    hn = np.concatenate([np.load(f"/data/neg/shard_00/chunk_{i:04d}_n.npy") for i in (28, 29)])
    hc = np.concatenate([np.load(f"/data/neg/shard_00/chunk_{i:04d}_ncnt.npy") for i in (28, 29)])
    metas = sorted(glob.glob("/data/agent/vec/*_meta.parquet"))
    tb = [pq.read_table(m, columns=["traj", "split", "config"]) for m in metas]
    vq, va, vd = [], [], []
    for m, r in zip(metas, tb):
        v = np.array(r.column("split").to_pylist()) == "val"
        if v.any():
            vq.append(np.load(m.replace("_meta.parquet", "_q.npy"))[v]); va.append(np.load(m.replace("_meta.parquet", "_a.npy"))[v])
            t = np.array(r.column("traj").to_pylist(), dtype=object)[v]; c = r.column("config")[0].as_py()
            vd.append(np.fromiter((zlib.crc32((c + x).encode()) for x in t), dtype=np.int64, count=len(t)))
    have_agent = len(vq) > 0
    if have_agent:
        vq, va, vd = np.concatenate(vq), np.concatenate(va), np.concatenate(vd)
    print(f"agent val {len(vq)}" if have_agent else "no agent data: agent test skipped", flush=True)

    @torch.no_grad()
    def pairs(sh, ah, s, q, a, d):
        L, hit, tot = [], 0, 0
        perm = np.random.default_rng(0).permutation(len(q))
        for i in range(0, len(perm) - 1023, 1024):
            b = np.sort(perm[i:i + 1024])
            zq, za = F.normalize(sh(T(q[b])), dim=-1), F.normalize(ah(T(a[b])), dim=-1)
            c = torch.from_numpy(d[b]).to(dev); same = c[None] == c[:, None]; same.fill_diagonal_(False)
            lg = (s * zq @ za.t()).masked_fill(same, float("-inf")); y = torch.arange(len(b), device=dev)
            L.append(((F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2).item()); hit += (lg.argmax(1) == y).sum().item(); tot += len(b)
        return round(float(np.mean(L)), 4), round(hit / tot, 4)

    @torch.no_grad()
    def tricky(sh, ah, s):
        hit = tot = 0; L = []
        for i in range(0, len(hq), 4096):
            j = min(i + 4096, len(hq)); c = torch.from_numpy(hc[i:j].astype(np.int64)).to(dev)
            zq, za = F.normalize(sh(T(hq[i:j])), dim=-1), F.normalize(ah(T(ha[i:j])), dim=-1)
            zn = F.normalize(ah(T(hn[i:j]).view(-1, DIM)), dim=-1).view(j - i, 10, -1)
            neg = torch.einsum("bh,bkh->bk", zq, zn).masked_fill(torch.arange(10, device=dev)[None] >= c[:, None], -9.0)
            pos = (zq * za).sum(-1, keepdim=True)
            v = c > 0; hit += (pos[v, 0] > neg[v].max(1).values).sum().item(); tot += v.sum().item()
            lg = (s * torch.cat([pos, neg], 1)).masked_fill(torch.cat([torch.zeros_like(pos, dtype=torch.bool), neg <= -9.0], 1), float("-inf"))
            L.append(F.cross_entropy(lg[v], torch.zeros(int(v.sum()), dtype=torch.long, device=dev)).item())
        return round(hit / tot, 4), round(float(np.mean(L)), 4), tot

    out = {}
    for path in heads:
        if path == "raw":
            sh = ah = nn.Identity(); scales, step = (10.0, 30.0, 100.0, 300.0, 1000.0), None
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sh, ah = head(), head(); sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
            scales, step = (torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0).item(),), ck.get("step")
        r = None
        for s in scales:  # one scale for a head; for "raw", keep the scale with the lowest general loss
            t, tl, n = tricky(sh, ah, s); gl, gt = pairs(sh, ah, s, gq, ga, gd)
            if r is None or gl < r["gen_loss"]:
                r = dict(tricky_top1=t, tricky_loss=tl, n=n, gen_loss=gl, gen_top1=gt, scale=round(s, 1), step=step)
        if have_agent:
            r["agent_loss"], r["agent_top1"] = pairs(sh, ah, r["scale"], vq, va, vd)
        out[path] = r
        print("RESULT", path, json.dumps(r), flush=True)
    json.dump(out, open(f"/data/runs/{out_name}.json", "w"), indent=1); vol.commit()


@app.local_entrypoint()
def main(heads: str, out: str = "final_eval"):
    run.remote(heads.split(","), out)

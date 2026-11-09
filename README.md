# clm-experiments

Training and evaluating an open System One model in the style of CLM-8B ("Contrastive Language Models", Kwok et al., Stanford and NVIDIA Research, 2026).

A System One model doesn't generate text. You give it a situation and a list of options, and it returns a probability for each option in one forward pass.

## The model

```text
Question + document -> Qwen3-8B (frozen) -> State head  -> 512-d
Answer              -> Qwen3-8B (frozen) -> Action head -> 512-d
```

Each head takes the last-token hidden state (4096-d), normalizes it, then runs 4096 -> 1536 -> 1536 (LayerNorm) -> 512 with GELU. The score is cosine similarity times a learned scale. Only the two heads are trained, about 18.9M parameters.

## Pipeline

| Stage | Data | Script |
|---|---|---|
| Data | [Q&A vectors](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron) (~1 TB), [hard negatives](https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings) (~0.25 TB) | `stage_data.py` |
| Pre-training | 621 chunks, InfoNCE, batch 16,384 | `train_heads.py --stage pre` |
| Mid-training | Pairs with up to 10 hard negatives | `train_heads.py --stage mid` |
| Post-training | Agent decisions from [ADP v2](https://huggingface.co/datasets/neulab/adp-v2), with replay | `build_agent_data.py`, `embed_agent.py`, `train_heads.py --stage post` |
| Verifier | DeepSWE steps | `run_deepswe.py` |
| Evaluation | Held-out Q&A shards, T-Rex | `final_eval.py`, `run_trex.py` |

## Run it

You need a [Modal](https://modal.com) account and about 1.3 TB of volume storage.

```bash
pip install modal
cd pipeline
modal run --detach stage_data.py
modal run --detach train_heads.py --stage pre --max-steps 30000
modal run --detach train_heads.py --stage mid --init /data/runs/pre/best.pt --run mid
modal run --detach build_agent_data.py
modal run --detach embed_agent.py
modal run --detach train_heads.py --stage post --init /data/runs/mid/best.pt --run post --max-steps 400 --lr 0.0003 --replay-negs
modal run run_deepswe.py --tag ours --init /data/runs/post/best.pt
modal run run_trex.py --tag ours --ckpt /data/runs/post/best.pt
```

Setup notes:
- Serving needs `vllm==0.11.0` with `transformers<5`.
- For DeepSWE evaluation, use the public `tarsur385/deepswe-prm-embeddings-8k` dataset.

## Targets

These are the published CLM-8B numbers we aim to match, then beat:

| Test | Published |
|---|---|
| Tricky Q&A, after pre-training | 52.1% |
| Tricky Q&A, after mid-training | 69.2% |
| Tricky Q&A, after post-training | 68.5% |
| DeepSWE, seed 1234 | 31/38 |
| T-Rex, median latency | 16.5 ms |
| T-Rex, courses survived, shield on | 5/5 |
| T-Rex, planner agreement, shield on | 65.8% |

## Results

None yet.

## Licence

Apache 2.0. See `LICENSE` and `NOTICE`.

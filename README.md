# clm-experiments

Training and evaluating an open System One model in the style of CLM-8B ("Contrastive Language Models", Kwok et al., Stanford and NVIDIA Research, 2026).

A System One model doesn't generate text. You give it a situation and a list of options, and it returns a probability for each option. The situation and each option go through the encoder once; no text is generated.

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
modal run --detach embed_agent_h100.py      # ~3.5x faster per token than embed_agent.py (L40S)
modal run --detach train_heads.py --stage post --init /data/runs/mid/best.pt --run post --max-steps 400 --lr 0.0003 --replay-negs
modal run final_eval.py --heads "raw,/data/heads/qwen3_8b_posttrained_head.pt,/data/runs/post/best.pt" --out post_compare
modal run run_deepswe.py --tag ours --init /data/runs/post/best.pt
modal run run_trex.py --tag ours --ckpt /data/runs/post/best.pt
```

Setup notes:
- Serving needs `vllm==0.11.0` with `transformers<5`.
- For DeepSWE evaluation, use the public `tarsur385/deepswe-prm-embeddings-8k` dataset.

## Targets

These are the published CLM-8B numbers we aim to match:

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

Run on Modal in November 2026, following the recipe in this README. "Authors" means the CLM authors' released heads
(`jackyk02/qwen3_8b_head`, `Contrastive-LM/CLM-v0.1-8B`), scored with the same code on the same held-out data.

| Test | Published | Authors' head, our code | Ours | |
|---|---|---|---|---|
| Tricky Q&A, after pre-training | 52.1% | 51.7% | 50.7% | 1.0 behind |
| Tricky Q&A, after mid-training | 69.2% | 69.2% | 68.5% | 0.7 behind |
| Tricky Q&A, after post-training | 68.5% | 68.6% | **69.0%** | ahead |
| DeepSWE, seed 1234 | 31/38 | 31/38 | 30/38 | |
| DeepSWE, mean of 5 seeds (random pick 28, perfect pick 34) | | 31.0/38 | **30.8/38** | tie |
| T-Rex, median decision latency in the 60 FPS game loop (L40S) | 16.5 ms | 17.0 ms | 17.1 ms | same |
| T-Rex, courses survived, shield on | 5/5 | 5/5 | 3/5 | behind |
| T-Rex, planner agreement, shield on | 65.8% | 65.3% | 2.7% | far behind |

Other numbers from the same runs:

| | Raw Qwen3-8B, no heads | Pre | Mid | Post |
|---|---|---|---|---|
| Tricky Q&A top-1 (191,161 questions, up to 10 look-alikes each) | 14.4% | 50.7% | 68.5% | 69.0% |
| General Q&A top-1 (200k pairs, 1,024-way) | 1.9% | 83.3% | 83.2% | 82.7% |
| Agent top-1 (our ADP v2 validation, 26,860 pairs, 1,024-way) | | | 0.5% | 19.6% |

### What we found

- **Pre-training**: 30,000 steps, 2.4 h on one L40S. `best.pt` is chosen by general loss, which kept step 30,000
  (50.7%) over step 28,000 (51.6%). The trainer now also keeps `best_hn.pt`, the best tricky-Q&A checkpoint.
  After step ~16,000 tricky-Q&A accuracy stayed flat while its loss rose, once the learned scale hit its cap of 100.
- **Mid-training**: 1,344 steps (8 passes), 57 min; it was still improving slowly at the end. The authors reached 69.2%
  in 420 steps. Most of our 0.7-point gap was inherited from pre-training.
- **Post-training**: 914,106 agent pairs from 39 ADP v2 configs (the six-steps-per-recording sample of
  `build_agent_data.py`), about 1.56B tokens embedded on H100s (~8.5 GPU-hours), then 400 steps in 7 min.
  Replaying hard negatives kept tricky Q&A at 69.0%.
- **DeepSWE** ties: the authors' head itself varies 30-32/38 across seeds. Each run first fine-tunes the head on
  DeepSWE's training tasks; 21 of the 38 held-out tasks have all 4 candidates passing and 4 have none, so only 13
  are decided by the pick (ours 9, theirs 10, random 7).
- **T-Rex** is the open gap. In a sweep of short games (2 per head, shield on), our mid head is as poor at the game as
  the authors' mid head (18-19% planner agreement); the authors' post-training lifts theirs to ~79%, ours drops to ~2%,
  so the gap most likely comes from post-training. (The full 5-course runs give 65.3% vs 2.7%.)
  Our post head almost never picks "run", the most common right answer. Two things we ruled out:
  wrapping the game state in the chat template does not help (25-27% offline either way), and embedding bare actions
  without narration (`embed_actions.py`, run `post_v2`) helps only a little (10% in the sweep).
  The authors' card for their 35B agent embeddings describes a different data recipe: plain-text states rendered as
  labelled paragraphs (`User:`, `Assistant:`, `Tool (name):`), actions as visible text plus `name: value` tool calls,
  every step of ~1M trajectories, plus Endless-Terminals and LiteCoder-Terminal traces. Their renderer and the 8B
  post-training data are not public, so matching T-Rex is the next experiment.

### Cost

About $82 on Modal: data staging ~$4, pre-training ~$13, mid-training ~$5, agent-pair build and H100 embedding ~$50,
post-training, evaluations, DeepSWE seeds and T-Rex runs ~$10.

### Changes to the pipeline made while running it

- `train_heads.py` also saves `best_hn.pt`, the checkpoint with the best tricky-Q&A top-1.
- `final_eval.py` adds a `raw` baseline (no heads), the tricky-Q&A loss, `--out`, and skips the agent test when no
  agent data exists.
- `build_agent_data.py` resumes broken downloads with HTTP Range requests (five configs of 0.5-7.7 GB failed
  mid-stream without it) and takes `--only`. Config names containing a dot (`allenai_Sera-4.6-Lite-T2`) are still skipped.
- `embed_agent.py` and `embed_agent_h100.py` take `--only`.

## Credits

The pipeline code is adapted from [VizuaraAI/system-one-from-scratch](https://github.com/VizuaraAI/system-one-from-scratch)
(Apache 2.0). The CLM method, data and reference heads are by Kwok et al. (Stanford and NVIDIA Research).

## Licence

Apache 2.0. See `LICENSE` and `NOTICE`.

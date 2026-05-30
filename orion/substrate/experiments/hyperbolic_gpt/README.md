# Hyperbolic GPT (research MVP)

Tiny decoder-only GPT with **hyperbolic geometry inside causal self-attention** — a quick experiment to see whether a Poincaré-ball distance penalty can train and produce coherent text.

## What this is

- Self-contained module under `orion/substrate/experiments/hyperbolic_gpt/`
- Hand-rolled Poincaré math (`hyperbolic.py`), no geoopt
- Attention logits: dot-product scores minus `λ * d_Poincaré(q_h, k_h)` before softmax
- Hyperbolic ops run in **fp32** inside attention; training may use fp16 AMP
- Optional **single-node DDP** via `torchrun` (`RANK` / `WORLD_SIZE` / `LOCAL_RANK`)

## What this is not

- Not a Docker service, FastAPI app, or Orion bus integration
- Not RAG, graph memory, Chroma, or recall plumbing
- Not a production package or Euclidean baseline (future work)

## Setup

```bash
pip install -r orion/substrate/experiments/hyperbolic_gpt/requirements.txt
```

Linux + CUDA recommended for training; smoke test runs on CPU.

## Commands

From repo root:

```bash
export PYTHONPATH=.

# Smoke test (forward, backward, short generate)
python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
bash orion/substrate/experiments/hyperbolic_gpt/scripts/smoke_test.sh

# Single-GPU / CPU training
python orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset text --text_path ./my_corpus.txt --device cuda --amp

# 2-GPU DDP TinyStories (default script)
bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh

# Generate from checkpoint
bash orion/substrate/experiments/hyperbolic_gpt/scripts/generate_sample.sh
```

### DDP training (exact)

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset tinystories \
  --out_dir ./runs/tinystories_hypgpt_4l_256d \
  --max_steps 10000 \
  --batch_size 32 \
  --block_size 256 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --lr 3e-4 \
  --eval_interval 250 \
  --save_interval 1000 \
  --amp
```

Per-process batch size is `--batch_size`; effective global batch ≈ `batch_size × nproc_per_node × grad_accum`.

Without `torchrun` env vars, `train.py` behaves as a normal single-process script.

## Target runs

1. Smoke test
2. 10k-step TinyStories (4 layers, 256 dim)
3. Later: ~125M-scale run and Euclidean comparison (not in v0)

## Warning

Research code — may be unstable. NaNs can appear if hyperbolic parameters diverge; ball projection and softplus on λ and curvature mitigate but do not eliminate risk.

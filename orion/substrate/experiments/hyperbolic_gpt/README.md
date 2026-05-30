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
- Not a production package

## v2: latent-semantic-manifold experiment

The v2 path keeps v1 intact and adds a paper-inspired manifold instrumentation layer:

- `model_v2.py` adds a switchable hyperbolic-attention path plus a low-rank `SemanticTangentAdapter` before the LM head.
- `manifold_metrics.py` adds training-time probes for Voronoi margin, normalized expressibility gap, entropy, a Fisher trace proxy, hidden norm, and tangent-energy.
- `train_v2.py` logs geometry during eval, so loss can be compared against margin / gap behavior instead of treating perplexity as the only signal.
- `generate_v2.py` loads v2 checkpoints.

The goal is not to prove the full latent-manifold theory inside this toy model. The goal is to make the theory operational: train a tiny GPT and observe whether manifold-aware components change loss, margin distributions, expressibility-gap fraction, entropy, and generation quality.

## Setup

```bash
pip install -r orion/substrate/experiments/hyperbolic_gpt/requirements.txt
```

Linux + CUDA recommended for training; smoke tests run on CPU.

## Commands

From repo root:

```bash
export PYTHONPATH=.

# v1 smoke test (forward, backward, short generate)
python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
bash orion/substrate/experiments/hyperbolic_gpt/scripts/smoke_test.sh

# v2 smoke test (adds manifold metrics)
python orion/substrate/experiments/hyperbolic_gpt/smoke_test_v2.py
bash orion/substrate/experiments/hyperbolic_gpt/scripts/smoke_test_v2.sh

# v1 single-GPU / CPU training
python orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset text --text_path ./my_corpus.txt --device cuda --amp

# v2 single-GPU / CPU training
python orion/substrate/experiments/hyperbolic_gpt/train_v2.py \
  --dataset text \
  --text_path ./my_corpus.txt \
  --out_dir ./runs/text_hypgpt_v2 \
  --device cuda \
  --semantic_adapter_rank 64 \
  --margin_gap_loss_weight 0.001 \
  --amp

# v1 2-GPU DDP TinyStories
bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh

# v2 2-GPU DDP TinyStories
bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_v2_tinystories.sh

# Generate from v1 checkpoint
bash orion/substrate/experiments/hyperbolic_gpt/scripts/generate_sample.sh

# Generate from v2 checkpoint
python orion/substrate/experiments/hyperbolic_gpt/generate_v2.py \
  --checkpoint ./runs/tinystories_hypgpt_v2_12l_768d \
  --prompt "Once upon a time" \
  --max_new_tokens 120 \
  --temperature 0.9 \
  --top_k 50
```

### v1 DDP training (exact)

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset tinystories \
  --out_dir ./runs/tinystories_hypgpt_4l_256d \
  --max_steps 10000 \
  --max_docs 50000 \
  --max_tokens 5000000 \
  --batch_size 16 \
  --block_size 256 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --lr 3e-4 \
  --eval_interval 250 \
  --save_interval 1000 \
  --amp
```

### v2 DDP training (Atlas/Circe-style larger run)

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  orion/substrate/experiments/hyperbolic_gpt/train_v2.py \
  --dataset tinystories \
  --out_dir ./runs/tinystories_hypgpt_v2_12l_768d \
  --max_steps 50000 \
  --max_docs 1000000 \
  --max_tokens 100000000 \
  --batch_size 4 \
  --grad_accum 4 \
  --block_size 256 \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --lr 3e-4 \
  --semantic_adapter_rank 128 \
  --margin_gap_loss_weight 0.001 \
  --margin_gap_epsilon 0.5 \
  --eval_interval 250 \
  --save_interval 1000 \
  --amp
```

Per-process batch size is `--batch_size`; effective global batch ≈ `batch_size × nproc_per_node × grad_accum`.

Without `torchrun` env vars, `train.py` and `train_v2.py` behave as normal single-process scripts.

### v2 metrics to watch

During `train_v2.py`, eval logs include:

- `loss_ce`: normal language-model cross entropy, without auxiliary loss terms.
- `gap@epsilon`: empirical normalized expressibility gap fraction, i.e. fraction of positions with top-1/top-2 margin below epsilon.
- `margin`: mean Voronoi margin.
- `entropy`: mean token entropy.
- `fisher_proxy`: cheap `1 - ||p||_2^2` distributional-spread proxy.
- `tangent`: adapter update energy relative to hidden-state energy.

A useful v2 run is one where `loss_ce` improves or matches v1 while `gap@0.5` decreases, `margin` rises moderately, entropy does not collapse, and tangent energy stays nonzero but small.

### Memory and eval notes

- Hyperbolic attention builds pairwise distances with broadcast shape `(B, H, T, T, D)` — much heavier than dot-product attention alone. On 16GB GPUs, prefer small per-process batches and use `--grad_accum`.
- `--max_docs` caps stories; `--max_tokens` caps the token list after encoding (stronger RAM bound).
- Under DDP, logged **eval loss** and v2 geometry are means of per-rank estimates on sharded eval tokens (`all_reduce`), not exact full-corpus estimates. Use for trend monitoring only.
- `--no_hyperbolic_attention` gives a Euclidean-ish control while preserving the semantic adapter and manifold metrics.

## Target runs

1. v1 smoke test
2. v2 smoke test
3. v1 10k-step TinyStories (4 layers, 256 dim)
4. v2 matched 10k-step TinyStories (same shape, adapter rank 64)
5. v2 12-layer / 768-dim run with gradient accumulation
6. Euclidean-control v2 run using `--no_hyperbolic_attention`

## Warning

Research code — may be unstable. NaNs can appear if hyperbolic parameters diverge; ball projection and softplus on λ and curvature mitigate but do not eliminate risk. Auxiliary margin-gap loss can also damage uncertainty if overweighted; start tiny (`0.0005`–`0.001`) and compare against a no-auxiliary-loss v2 baseline.

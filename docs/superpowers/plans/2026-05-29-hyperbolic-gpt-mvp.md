# Hyperbolic GPT MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a self-contained research MVP at `orion/substrate/experiments/hyperbolic_gpt/` — a tiny decoder-only GPT with **hyperbolic distance inside causal self-attention** (not post-hoc analysis) to test whether a geometric inductive bias can train and generate coherent text.

**Architecture:** All code lives under one module folder with its own `requirements.txt`. Poincaré ball math is hand-rolled in `hyperbolic.py` (no geoopt). `HyperbolicCausalSelfAttention` computes dot-product scores, subtracts `softplus(λ) * d_poincaré(q_h, k_h)` per head, then softmax — hyperbolic ops run in **fp32** inside the attention path while the rest of training may use fp16 AMP. Training uses GPT-2 tokenizer + TinyStories (or local text fallback); checkpoints save `model.pt`, `config.json`, and tokenizer metadata for `generate.py`. Optional **single-node DDP** via `torchrun` (env `RANK` / `WORLD_SIZE` / `LOCAL_RANK`); plain `python train.py` unchanged when those are absent.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face `transformers` + `datasets`, `tqdm`, `numpy`; optional `safetensors`. Linux + CUDA (V100-class), mixed precision training.

**Explicit non-goals (do not implement in this pass):** Docker service, FastAPI, RAG/graph memory, bus events, `orion/bus/channels.yaml`, `orion/schemas/registry.py`, root `requirements.txt`, docker-compose, service `settings.py`, Euclidean comparison model, benchmark harness, **FSDP**, **DeepSpeed**, custom multi-node launchers (only what `torchrun` provides). **Do not edit files outside** `orion/substrate/experiments/` except the plan doc and PR report. **No `.env` sync** — this experiment has no service env; dependencies install from the module-local `requirements.txt` only.

**Critical invariant:** Hyperbolic distance must be computed **inside** `HyperbolicCausalSelfAttention.forward()` and subtracted from attention logits **before** `softmax`. Never compute distance only in logging or analysis hooks.

**Worktree:** Implement on branch `feat/hyperbolic-gpt-mvp` in `.worktrees/feat-hyperbolic-gpt-mvp` (project-local worktrees; already gitignored).

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/substrate/experiments/__init__.py` | Package marker for `experiments` (plural; distinct from existing `orion/substrate/experiment/`) |
| `orion/substrate/experiments/hyperbolic_gpt/__init__.py` | Re-exports version string only |
| `orion/substrate/experiments/hyperbolic_gpt/requirements.txt` | torch, transformers, datasets, tqdm, numpy |
| `orion/substrate/experiments/hyperbolic_gpt/config.py` | `HyperbolicGPTConfig` dataclass + JSON serde |
| `orion/substrate/experiments/hyperbolic_gpt/hyperbolic.py` | Poincaré ball primitives (fp32-safe) |
| `orion/substrate/experiments/hyperbolic_gpt/model.py` | `HyperbolicGPT`, blocks, **hyperbolic causal attention** |
| `orion/substrate/experiments/hyperbolic_gpt/data.py` | GPT-2 tokenizer, TinyStories / text dataset |
| `orion/substrate/experiments/hyperbolic_gpt/train.py` | Training loop, AMP, checkpoints; optional single-node DDP (`torchrun`) |
| `orion/substrate/experiments/hyperbolic_gpt/generate.py` | Autoregressive sampling from checkpoint |
| `orion/substrate/experiments/hyperbolic_gpt/smoke_test.py` | Tiny forward/backward/generate; assert no NaNs |
| `orion/substrate/experiments/hyperbolic_gpt/README.md` | What this is / is not, commands, warnings |
| `orion/substrate/experiments/hyperbolic_gpt/scripts/smoke_test.sh` | Wrapper for smoke test |
| `orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh` | 10k-step TinyStories job via `torchrun` (2 GPUs) |
| `orion/substrate/experiments/hyperbolic_gpt/scripts/generate_sample.sh` | Generate from trained checkpoint |

**Run context:** All commands assume repo root `/mnt/scripts/Orion-Sapienform` with `PYTHONPATH=.` (or scripts insert repo root into `sys.path` like `orion/substrate/scripts/smoke_mutation_v21.py`).

---

### Task 0: Git worktree and branch

**Files:** (none in repo — git only)

- [ ] **Step 1: Create branch and worktree**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main 2>/dev/null || true
git branch feat/hyperbolic-gpt-mvp origin/main 2>/dev/null || git branch feat/hyperbolic-gpt-mvp main
git worktree add .worktrees/feat-hyperbolic-gpt-mvp feat/hyperbolic-gpt-mvp
cd .worktrees/feat-hyperbolic-gpt-mvp
```

Expected: worktree path exists; `git branch --show-current` → `feat/hyperbolic-gpt-mvp`.

- [ ] **Step 2: Create venv for experiment (optional but recommended)**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
python3 -m venv .venv-hyperbolic-gpt
source .venv-hyperbolic-gpt/bin/activate
pip install -U pip
# requirements.txt created in Task 1; install after Task 1
```

- [ ] **Step 3: Commit plan only (if plan added from main workspace)**

```bash
git add docs/superpowers/plans/2026-05-29-hyperbolic-gpt-mvp.md
git commit -m "docs: add hyperbolic GPT MVP implementation plan"
```

---

### Task 1: Package scaffold and requirements

**Files:**
- Create: `orion/substrate/experiments/__init__.py`
- Create: `orion/substrate/experiments/hyperbolic_gpt/__init__.py`
- Create: `orion/substrate/experiments/hyperbolic_gpt/requirements.txt`

- [ ] **Step 1: Create package markers**

`orion/substrate/experiments/__init__.py`:

```python
"""Substrate research experiments (self-contained; not production services)."""
```

`orion/substrate/experiments/hyperbolic_gpt/__init__.py`:

```python
"""Hyperbolic-attention decoder GPT research MVP."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Create requirements.txt**

`orion/substrate/experiments/hyperbolic_gpt/requirements.txt`:

```text
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
tqdm>=4.66.0
numpy>=1.24.0
```

- [ ] **Step 3: Install deps in worktree venv**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
source .venv-hyperbolic-gpt/bin/activate
pip install -r orion/substrate/experiments/hyperbolic_gpt/requirements.txt
```

- [ ] **Step 4: Commit**

```bash
git add orion/substrate/experiments/
git commit -m "feat(substrate): scaffold hyperbolic_gpt experiment package"
```

---

### Task 2: Config dataclass

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/config.py`

- [ ] **Step 1: Implement `HyperbolicGPTConfig`**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class HyperbolicGPTConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    geo_lambda_init: float = 0.05
    curvature_init: float = 1.0
    use_learned_curvature: bool = True
    use_learned_geo_lambda: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperbolicGPTConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> HyperbolicGPTConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
```

- [ ] **Step 2: Quick import check**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
PYTHONPATH=. python -c "from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig; c=HyperbolicGPTConfig(); assert c.head_dim==64"
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/config.py
git commit -m "feat(hyperbolic_gpt): add HyperbolicGPTConfig"
```

---

### Task 3: Poincaré math (`hyperbolic.py`)

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/hyperbolic.py`

- [ ] **Step 1: Implement primitives (full file)**

```python
from __future__ import annotations

import torch


def _eps(x: torch.Tensor, eps: float) -> float:
    return eps


def project_to_ball(x: torch.Tensor, c: torch.Tensor | float, eps: float = 1e-5) -> torch.Tensor:
    """Project ambient vectors into the open Poincaré ball (||x|| < 1/sqrt(c))."""
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    max_norm = (1.0 - eps) / torch.sqrt(c_t)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    factor = torch.clamp(max_norm / norm, max=1.0)
    return x * factor


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor | float, eps: float = 1e-5) -> torch.Tensor:
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    den = 1.0 + 2.0 * c_t * xy + c_t**2 * x2 * y2
    out = num / den.clamp_min(eps)
    return project_to_ball(out, c_t, eps=eps)


def expmap0(v: torch.Tensor, c: torch.Tensor | float, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at origin: tangent vector -> point on ball."""
    c_t = torch.as_tensor(c, dtype=v.dtype, device=v.device)
    sqrt_c = torch.sqrt(c_t)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
    # tanh(sqrt(c)/2 * ||v||) / (sqrt(c) * ||v||) * v  — standard exp_0
    coef = torch.tanh(sqrt_c * 0.5 * v_norm) / (sqrt_c * v_norm)
    y = coef * v
    return project_to_ball(y, c_t, eps=eps)


def _artanh_clamped(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    d_c(x, y) = 2/sqrt(c) * artanh( sqrt(c) * || (-x) ⊕_c y || )
    x, y: (..., D) matching batch dims
    returns: (...) scalar distance per pair
    """
    c_t = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    x_b = project_to_ball(x, c_t, eps=eps)
    y_b = project_to_ball(y, c_t, eps=eps)
    minus_x = -x_b
    diff = mobius_add(minus_x, y_b, c_t, eps=eps)
    diff_norm = torch.linalg.vector_norm(diff, dim=-1).clamp_min(eps)
    arg = torch.sqrt(c_t) * diff_norm
    arg = arg.clamp(max=1.0 - eps)
    return (2.0 / torch.sqrt(c_t)) * _artanh_clamped(arg, eps=eps)


def poincare_distance_pairs(
    q: torch.Tensor,
    k: torch.Tensor,
    c: torch.Tensor | float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Pairwise distances for attention.
    q: (B, H, Tq, D), k: (B, H, Tk, D) -> (B, H, Tq, Tk)
    """
    # q_i vs k_j for all i,j
    qi = q.unsqueeze(-2)  # B,H,Tq,1,D
    kj = k.unsqueeze(-3)  # B,H,1,Tk,D
    diff = mobius_add(-project_to_ball(qi, c, eps), project_to_ball(kj, c, eps), c, eps)
    # Use full formula on broadcasted pairs
    c_t = torch.as_tensor(c, dtype=q.dtype, device=q.device)
    diff_norm = torch.linalg.vector_norm(diff, dim=-1).clamp_min(eps)
    arg = (torch.sqrt(c_t) * diff_norm).clamp(max=1.0 - eps)
    return (2.0 / torch.sqrt(c_t)) * _artanh_clamped(arg, eps)
```

**Note for implementer:** `poincare_distance_pairs` can call `mobius_add(-qi, kj, c)` directly on broadcast tensors; verify shapes `(B,H,T,T)` with a one-liner in REPL before commit.

- [ ] **Step 2: Sanity check in REPL**

```bash
PYTHONPATH=. python - <<'PY'
import torch
from orion.substrate.experiments.hyperbolic_gpt.hyperbolic import expmap0, poincare_distance
q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
qh = expmap0(q, 1.0)
kh = expmap0(k, 1.0)
d = poincare_distance(qh, kh, 1.0)
assert torch.isfinite(d).all()
print("ok", d.shape)
PY
```

- [ ] **Step 3: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/hyperbolic.py
git commit -m "feat(hyperbolic_gpt): add Poincaré ball math primitives"
```

---

### Task 4: Model and hyperbolic attention (`model.py`)

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/model.py`

**Invariant checklist for this file:**
- `HyperbolicCausalSelfAttention.forward` computes `attn_logits = dot_scores - lambda_geo * dist` then causal mask then softmax.
- `q_h`, `k_h` from dedicated `q_geo`/`k_geo` linear layers + `expmap0` in fp32.
- `lambda_geo` and `c` positive via `softplus` on parameters or buffers.

- [ ] **Step 1: Implement model (full file — key sections below; implement complete file in repo)**

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.hyperbolic import (
    expmap0,
    poincare_distance_pairs,
)


class HyperbolicCausalSelfAttention(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Hyperbolic pathway (separate from dot-product q/k)
        self.q_geo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_geo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

        if config.use_learned_geo_lambda:
            self.lambda_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(config.geo_lambda_init)))
            )
        else:
            self.register_buffer(
                "lambda_raw",
                torch.tensor(math.log(math.expm1(config.geo_lambda_init))),
            )

        if config.use_learned_curvature:
            self.curvature_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(config.curvature_init)))
            )
        else:
            self.register_buffer(
                "curvature_raw",
                torch.tensor(math.log(math.expm1(config.curvature_init))),
            )

    @property
    def geo_lambda(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.curvature_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        H, D = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # B,H,T,D
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Euclidean dot-product scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))  # B,H,T,T

        # Hyperbolic distance penalty (fp32 inside attention path)
        qg = self.q_geo(x).view(B, T, H, D).transpose(1, 2)
        kg = self.k_geo(x).view(B, T, H, D).transpose(1, 2)
        c = self.curvature
        lam = self.geo_lambda
        with torch.autocast(device_type=q.device.type, enabled=False):
            qg32 = qg.float()
            kg32 = kg.float()
            qh = expmap0(qg32, c.item() if c.numel() == 1 else c, eps=1e-5)
            kh = expmap0(kg32, c.item() if c.numel() == 1 else c, eps=1e-5)
            dist = poincare_distance_pairs(qh, kh, c, eps=1e-5)  # B,H,T,T

        att = att - lam * dist.to(dtype=att.dtype)

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = HyperbolicCausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HyperbolicGPT(nn.Module):
    def __init__(self, config: HyperbolicGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(0, T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
```

- [ ] **Step 2: Shape-only forward (no train yet)**

```bash
PYTHONPATH=. python - <<'PY'
import torch
from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.model import HyperbolicGPT
cfg = HyperbolicGPTConfig(n_layer=2, n_head=2, n_embd=128, block_size=64, vocab_size=1000)
m = HyperbolicGPT(cfg)
x = torch.randint(0, 1000, (2, 32))
logits, loss = m(x, x)
assert logits.shape == (2, 32, 1000)
assert torch.isfinite(logits).all()
print("ok")
PY
```

- [ ] **Step 3: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/model.py
git commit -m "feat(hyperbolic_gpt): decoder GPT with hyperbolic causal attention"
```

---

### Task 5: Data loading (`data.py`)

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/data.py`

- [ ] **Step 1: Implement dataset helpers**

```python
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


def load_gpt2_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TokenBlockIterable(IterableDataset):
    """Streams token blocks (x, y) with y = x shifted by 1."""

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        stride: int | None = None,
    ) -> None:
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride or block_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        data = self.token_ids
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Not enough tokens for one block; provide longer text.")
        starts = list(range(0, max_start, self.stride))
        random.shuffle(starts)
        for i in starts:
            chunk = data[i : i + self.block_size + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y


def load_tinystories_tokens(max_docs: int | None = None) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tok = load_gpt2_tokenizer()
    ids: list[int] = []
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = row.get("text") or row.get("story") or ""
        ids.extend(tok.encode(text + tok.eos_token))
    if len(ids) < 256:
        raise RuntimeError("TinyStories load produced too few tokens")
    return ids


def load_text_file_tokens(path: str | Path) -> list[int]:
    tok = load_gpt2_tokenizer()
    text = Path(path).read_text(encoding="utf-8")
    return tok.encode(text)


def build_dataloader(
    token_ids: list[int],
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Simple batching iterator (not DataLoader) for research MVP."""

    def _batch_iter() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buf_x: list[torch.Tensor] = []
        buf_y: list[torch.Tensor] = []
        for x, y in TokenBlockIterable(token_ids, block_size):
            buf_x.append(x)
            buf_y.append(y)
            if len(buf_x) >= batch_size:
                yield (
                    torch.stack(buf_x).to(device),
                    torch.stack(buf_y).to(device),
                )
                buf_x, buf_y = [], []

    return _batch_iter()
```

- [ ] **Step 2: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/data.py
git commit -m "feat(hyperbolic_gpt): GPT-2 tokenizer and TinyStories blocks"
```

---

### Task 6: Training (`train.py`)

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/train.py`

- [ ] **Step 1: Implement argparse + training loop**

Implement `main()` with arguments exactly as spec:

| Flag | Default |
|------|---------|
| `--dataset` | required: `tinystories` or `text` |
| `--text_path` | None |
| `--out_dir` | `./runs/hyperbolic_gpt` |
| `--max_steps` | 1000 |
| `--batch_size` | 16 |
| `--block_size` | 256 |
| `--n_layer` | 4 |
| `--n_head` | 4 |
| `--n_embd` | 256 |
| `--lr` | 3e-4 |
| `--grad_accum` | 1 |
| `--eval_interval` | 100 |
| `--save_interval` | 500 |
| `--device` | cuda if available else cpu |
| `--compile` | false |
| `--amp` | true |

Training loop requirements:
- AdamW, grad clip 1.0
- `torch.cuda.amp.autocast` + `GradScaler` when `--amp` and cuda
- Print loss every 10 steps
- Eval: held-out slice of tokens (last 5% of stream) average loss every `eval_interval`
- Save checkpoint dir: `model.pt` (state_dict + optimizer optional), `config.json`, `meta.json` with `tokenizer: gpt2`

Skeleton for checkpoint save:

```python
meta = {"tokenizer": "gpt2", "block_size": config.block_size}
torch.save({"model": model.state_dict(), "config": config.to_dict()}, out / "model.pt")
config.save_json(out / "config.json")
Path(out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
```

Insert repo root at top of `train.py`:

```python
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 2: Dry-run 2 steps on CPU (fast sanity)**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
PYTHONPATH=. python orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset text \
  --text_path /tmp/hypgpt_smoke.txt \
  --max_steps 2 \
  --batch_size 2 \
  --block_size 64 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 128 \
  --device cpu \
  --amp false \
  --out_dir /tmp/hypgpt_run
```

Create `/tmp/hypgpt_smoke.txt` first: `echo "Once upon a time there was a small robot." > /tmp/hypgpt_smoke.txt`

Expected: loss prints; no NaN; checkpoint files under `/tmp/hypgpt_run`.

- [ ] **Step 3: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/train.py
git commit -m "feat(hyperbolic_gpt): training loop with AMP and checkpoints"
```

---

### Task 6b: Single-node DDP in `train.py`

**Files:**
- Modify: `orion/substrate/experiments/hyperbolic_gpt/train.py`
- Modify: `orion/substrate/experiments/hyperbolic_gpt/data.py` (optional small helper for rank sharding)

**Behavior contract:**
- If `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` are **not** all set → identical to Task 6 single-process path (CPU or one CUDA device).
- If all three are set → DDP active: `nccl`, `cuda:LOCAL_RANK`, `DistributedDataParallel`, rank-0-only logs/checkpoints, `model.module.state_dict()` on save.
- No FSDP, no DeepSpeed, no extra multi-node wiring beyond `torchrun`.

- [ ] **Step 1: Add DDP helpers at top of `train.py` (after imports)**

```python
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_setup() -> tuple[bool, int, int, int]:
    """Return (is_ddp, rank, world_size, local_rank)."""
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if not all(k in os.environ for k in keys):
        return False, 0, 1, 0
    return (
        True,
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
    )


def ddp_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def shard_token_ids(token_ids: list[int], rank: int, world_size: int) -> list[int]:
    """Rank-aware partition for IterableDataset (no DistributedSampler)."""
    return token_ids[rank::world_size]
```

- [ ] **Step 2: Wire setup in `main()` before model creation**

```python
is_ddp, rank, world_size, local_rank = ddp_setup()
if is_ddp:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA (torchrun + nccl)")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device(args.device)  # existing argparse default logic
```

- [ ] **Step 3: Shard tokens and wrap model**

After loading `token_ids` (train split), before the batch iterator:

```python
train_ids = token_ids  # or all but eval holdout
if is_ddp:
    train_ids = shard_token_ids(train_ids, rank, world_size)
```

After `model = HyperbolicGPT(config).to(device)`:

```python
if is_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

Replace all bare `print(...)` loss/logging with `ddp_print(rank, ...)`.

- [ ] **Step 4: Checkpoint save (rank 0 only, unwrap DDP)**

```python
def save_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    config: HyperbolicGPTConfig,
    rank: int,
) -> None:
    if rank != 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    meta = {"tokenizer": "gpt2", "block_size": config.block_size}
    torch.save({"model": raw.state_dict(), "config": config.to_dict()}, out_dir / "model.pt")
    config.save_json(out_dir / "config.json")
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
```

- [ ] **Step 5: Eval loss — all-reduce mean across ranks**

On eval steps, each rank computes `eval_loss` on its shard (same held-out tail logic as Task 6, applied per-rank or only on rank-0 eval slice — **prefer all ranks eval on local shard then reduce**):

```python
def reduce_mean_scalar(value: float, device: torch.device, is_ddp: bool) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float64)
    if is_ddp:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())
```

Call after eval forward; log `reduce_mean_scalar(eval_loss, device, is_ddp)` only via `ddp_print`.

**Alternative (acceptable):** eval only on rank 0 with full eval token slice — document in README. Prefer all-reduce for parity with training shards.

- [ ] **Step 6: Shutdown barrier**

At end of `main()`:

```python
if is_ddp:
    dist.barrier()
    dist.destroy_process_group()
```

- [ ] **Step 7: Dry-run DDP 2 steps (2 GPUs, if available)**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
export PYTHONPATH=.
echo "Once upon a time there was a small robot." > /tmp/hypgpt_smoke.txt
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  orion/substrate/experiments/hyperbolic_gpt/train.py \
  --dataset text \
  --text_path /tmp/hypgpt_smoke.txt \
  --max_steps 2 \
  --batch_size 2 \
  --block_size 64 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 128 \
  --amp \
  --out_dir /tmp/hypgpt_ddp_run
```

Expected: rank 0 prints losses; `/tmp/hypgpt_ddp_run/model.pt` exists; no NCCL hang at exit.

If only one GPU is available, skip this step and note `UNVERIFIED` for DDP in PR — single-GPU `python train.py` must still pass Task 6 Step 2.

- [ ] **Step 8: Run smoke test (DDP does not affect smoke_test.py)**

```bash
PYTHONPATH=. python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
```

Expected: `SMOKE TEST PASSED`

- [ ] **Step 9: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/train.py
git add orion/substrate/experiments/hyperbolic_gpt/data.py  # only if helper added
git commit -m "feat(hyperbolic_gpt): single-node DDP via torchrun"
```

**Exact DDP TinyStories command (document in README + PR):**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
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

**Single-GPU / CPU (unchanged):**

```bash
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/train.py --dataset tinystories --device cuda --amp ...
```

---

### Task 7: Generation (`generate.py`)

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/generate.py`

- [ ] **Step 1: Implement CLI**

Arguments: `--checkpoint` (dir with model.pt), `--prompt`, `--max_new_tokens` (default 100), `--temperature` (1.0), `--top_k` (optional), `--device`.

Load `HyperbolicGPTConfig.load_json`, build model, load weights, encode prompt with GPT-2, call `model.generate`, decode and print.

- [ ] **Step 2: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/generate.py
git commit -m "feat(hyperbolic_gpt): checkpoint sampling CLI"
```

---

### Task 8: Smoke test (`smoke_test.py`)

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/smoke_test.py`

- [ ] **Step 1: Implement smoke test**

```python
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.substrate.experiments.hyperbolic_gpt.config import HyperbolicGPTConfig
from orion.substrate.experiments.hyperbolic_gpt.model import HyperbolicGPT


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HyperbolicGPTConfig(
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=64,
    )
    model = HyperbolicGPT(cfg).to(device)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    logits, loss = model(x, y)
    assert torch.isfinite(logits).all(), "non-finite logits"
    assert loss is not None and torch.isfinite(loss), "non-finite loss"
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"
    model.eval()
    gen = model.generate(x[:1, :8], max_new_tokens=5, temperature=1.0)
    assert torch.isfinite(gen.float()).all(), "non-finite generation"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
PYTHONPATH=. python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
```

Expected stdout contains: `SMOKE TEST PASSED`

- [ ] **Step 3: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
git commit -m "test(hyperbolic_gpt): add smoke test for forward/backward/generate"
```

---

### Task 9: Shell scripts

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/scripts/smoke_test.sh`
- Create: `orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh`
- Create: `orion/substrate/experiments/hyperbolic_gpt/scripts/generate_sample.sh`

- [ ] **Step 1: `scripts/smoke_test.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
```

- [ ] **Step 2: `scripts/train_tinystories.sh` (2-GPU DDP via torchrun)**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
TRAIN="${ROOT}/orion/substrate/experiments/hyperbolic_gpt/train.py"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  "${TRAIN}" \
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

Per-process batch size is `--batch_size`; global effective batch ≈ `batch_size * nproc_per_node * grad_accum`.

- [ ] **Step 3: `scripts/generate_sample.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/generate.py \
  --checkpoint ./runs/tinystories_hypgpt_4l_256d \
  --prompt "Once upon a time" \
  --max_new_tokens 120 \
  --temperature 0.9 \
  --top_k 40
```

- [ ] **Step 4: chmod +x**

```bash
chmod +x orion/substrate/experiments/hyperbolic_gpt/scripts/*.sh
```

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/scripts/
git commit -m "chore(hyperbolic_gpt): add smoke, train, and generate shell scripts"
```

---

### Task 10: README

**Files:**
- Create: `orion/substrate/experiments/hyperbolic_gpt/README.md`

- [ ] **Step 1: Write README**

Include sections:
1. **What this is** — research MVP; hyperbolic distance in attention logits.
2. **What this is not** — not a service, not RAG, not Docker, not bus-integrated.
3. **Attention modification** — `softmax((QK^T / sqrt(d)) - λ * d_P(q_h, k_h))` with `q_h,k_h` from `expmap0` on geo projections; fp32 hyperbolic math.
4. **Setup** — `pip install -r requirements.txt`, CUDA optional for smoke CPU.
5. **Commands:**

```bash
# from repo root
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh
bash orion/substrate/experiments/hyperbolic_gpt/scripts/generate_sample.sh
```

6. **Target runs** — smoke → 10k TinyStories (DDP script) → future 125M-scale (not this PR).
7. **Distributed training** — single-node only: `torchrun --standalone --nproc_per_node=2` sets `RANK` / `WORLD_SIZE` / `LOCAL_RANK`; plain `python train.py` for one GPU or CPU. No FSDP/DeepSpeed.
8. **Warning** — unstable research code; NaNs possible if curvature/lambda explode (softplus + ball projection mitigate).

- [ ] **Step 2: Commit**

```bash
git add orion/substrate/experiments/hyperbolic_gpt/README.md
git commit -m "docs(hyperbolic_gpt): add README for hyperbolic attention MVP"
```

---

### Task 11: Final verification, code review, PR

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-29-hyperbolic-gpt-mvp-pr.md`

- [ ] **Step 1: Run smoke test (required evidence)**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-hyperbolic-gpt-mvp
PYTHONPATH=. python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
bash orion/substrate/experiments/hyperbolic_gpt/scripts/smoke_test.sh
```

Expected: `SMOKE TEST PASSED` (smoke test does not use DDP).

- [ ] **Step 1b: Record exact DDP training command in PR report**

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

Optional: run `bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh` on a 2-GPU host.

- [ ] **Step 2: Code review subagent**

**REQUIRED SUB-SKILL:** `superpowers:requesting-code-review`

Dispatch subagent with:
- Base: `origin/main`
- Head: `feat/hyperbolic-gpt-mvp`
- Focus: hyperbolic distance **inside** `HyperbolicCausalSelfAttention.forward`; fp32 geo ops; DDP saves `unwrap(model).state_dict()`; rank-0-only checkpoints; no service bleed; file boundary respected.

Fix all substantive issues before PR.

- [ ] **Step 3: Write PR report**

`docs/superpowers/pr-reports/2026-05-29-hyperbolic-gpt-mvp-pr.md` with Summary, Files table, Test plan, TinyStories command:

```bash
export PYTHONPATH=.
bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh
```

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/hyperbolic-gpt-mvp
gh pr create --title "feat(substrate): hyperbolic GPT attention MVP" --body "$(cat <<'EOF'
## Summary
- Adds self-contained research experiment `orion/substrate/experiments/hyperbolic_gpt/`
- Decoder-only GPT with Poincaré distance penalty inside causal self-attention (fp32 geo math, AMP elsewhere)
- Includes smoke test, TinyStories training script, and generation CLI

## Test plan
- [ ] `PYTHONPATH=. python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py`
- [ ] Optional: 2-GPU DDP TinyStories via `scripts/train_tinystories.sh` or torchrun command in PR report

EOF
)"
```

---

## Self-review

### 1. Spec coverage

| Requirement | Task |
|-------------|------|
| All files under `hyperbolic_gpt/` | Tasks 1–10 |
| No Docker/service/bus | Header non-goals |
| Poincaré math manual | Task 3 |
| Hyperbolic distance in attention path | Task 4 (invariant) |
| Config defaults | Task 2 |
| train.py args + AMP | Task 6 |
| Single-node DDP (torchrun) | Task 6b |
| DDP train script (2 GPU) | Task 9 |
| TinyStories + text fallback | Task 5–6 |
| generate.py | Task 7 |
| smoke_test.py | Task 8 |
| Shell scripts | Task 9 |
| README | Task 10 |
| smoke_test command | Tasks 8, 11 |
| Worktree + commits | Task 0, each commit step |
| Code review + PR | Task 11 |

**Out of scope by design:** `.env`, `settings.py`, docker-compose, bus/registry — user paste template does not apply to this isolated experiment.

### 2. Placeholder scan

No TBD steps; each task includes concrete paths and code.

### 3. Type consistency

- `HyperbolicGPTConfig.head_dim` used in attention
- Checkpoint `config.json` matches `HyperbolicGPTConfig.load_json`
- `geo_lambda` / `curvature` via `softplus` everywhere

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-29-hyperbolic-gpt-mvp.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks (**REQUIRED SUB-SKILL:** `superpowers:subagent-driven-development`)

2. **Inline Execution** — run tasks in this session with checkpoints (**REQUIRED SUB-SKILL:** `superpowers:executing-plans`)

**Which approach?**

After implementation:

**Smoke test** (from repo root):

```bash
export PYTHONPATH=.
python orion/substrate/experiments/hyperbolic_gpt/smoke_test.py
```

**TinyStories DDP training** (2 GPUs, from repo root):

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

Or: `bash orion/substrate/experiments/hyperbolic_gpt/scripts/train_tinystories.sh`

**Single-GPU** (no torchrun): `python orion/substrate/experiments/hyperbolic_gpt/train.py --dataset tinystories --device cuda --amp ...`

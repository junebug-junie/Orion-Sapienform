from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer


def load_gpt2_tokenizer() -> PreTrainedTokenizer:
    from transformers import AutoTokenizer

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


def _truncate_tokens(ids: list[int], max_tokens: int | None) -> list[int]:
    if max_tokens is not None and len(ids) > max_tokens:
        return ids[:max_tokens]
    return ids


def load_tinystories_tokens(
    max_docs: int | None = None,
    max_tokens: int | None = None,
) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tok = load_gpt2_tokenizer()
    ids: list[int] = []
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = row.get("text") or row.get("story") or ""
        ids.extend(tok.encode(text + tok.eos_token))
        if max_tokens is not None and len(ids) >= max_tokens:
            ids = ids[:max_tokens]
            break
    if len(ids) < 256:
        raise RuntimeError("TinyStories load produced too few tokens")
    return ids


def load_text_file_tokens(path: str | Path, max_tokens: int | None = None) -> list[int]:
    tok = load_gpt2_tokenizer()
    text = Path(path).read_text(encoding="utf-8")
    return _truncate_tokens(tok.encode(text + tok.eos_token), max_tokens)


def shard_token_ids(token_ids: list[int], rank: int, world_size: int) -> list[int]:
    """Rank-aware partition for IterableDataset training shards."""
    return token_ids[rank::world_size]


def iter_batches(
    token_ids: list[int],
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    buf_x: list[torch.Tensor] = []
    buf_y: list[torch.Tensor] = []
    for x, y in TokenBlockIterable(token_ids, block_size):
        buf_x.append(x)
        buf_y.append(y)
        if len(buf_x) >= batch_size:
            yield torch.stack(buf_x).to(device), torch.stack(buf_y).to(device)
            buf_x, buf_y = [], []

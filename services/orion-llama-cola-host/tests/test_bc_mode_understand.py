from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig

from intention import IntentionForCausalLM


def _tiny_config() -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    # CoLA-specific fields required by IntentionModel_v1.__init__.
    config.v = 1
    config.num_code = 5
    config.num_action_layer = 1
    config.num_policy_layer = 1
    config.num_dyna_layer = 1
    return config


def _tiny_model() -> IntentionForCausalLM:
    torch.manual_seed(0)
    model = IntentionForCausalLM(_tiny_config())
    model.eval()
    return model


def test_bc_mode_returns_three_tensors_with_valid_distribution():
    model = _tiny_model()
    input_ids = torch.randint(low=1, high=37, size=(1, 6))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        policy_logits, action_idx, action_probs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            bc_mode=True,
        )

    assert action_probs.shape == (1, 6, 5)
    assert action_idx.shape == (1, 6)
    assert policy_logits.shape == (1, 6, 5)

    # A real softmax distribution over the 5-code action codebook.
    row_sums = action_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(action_probs >= 0.0)


def test_bc_mode_is_deterministic_across_repeated_calls():
    model = _tiny_model()
    input_ids = torch.randint(low=1, high=37, size=(1, 6))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        _, _, probs_1 = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, bc_mode=True
        )
        _, _, probs_2 = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, bc_mode=True
        )

    # Unlike the stochastic policy-sampling path (tau=2.0 top-k sampling), the
    # bc_mode/IDM branch has no sampling in it -- same input must give the same
    # distribution every time.
    assert torch.allclose(probs_1, probs_2)


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        ids = [((abs(hash(w)) % 36) + 1) for w in text.split()] or [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return _FakeBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


def test_understand_endpoint_pools_distribution_over_tokens(monkeypatch):
    from app import main as app_main

    model = _tiny_model()
    monkeypatch.setattr(app_main.state, "llm", model)
    monkeypatch.setattr(app_main.state, "tokenizer", _FakeTokenizer())
    monkeypatch.setattr(app_main.state, "device", torch.device("cpu"))

    response = app_main.understand(app_main.UnderstandRequest(text="hello there world", doc_id="turn-1"))

    assert response.doc_id == "turn-1"
    assert response.embedding_dim == 5
    assert len(response.embedding) == 5
    assert response.token_count == 3
    # Pooled mean of per-token softmax rows is itself ~a distribution.
    assert sum(response.embedding) == pytest.approx(1.0, abs=1e-4)


def test_understand_endpoint_rejects_empty_text(monkeypatch):
    from fastapi import HTTPException
    from app import main as app_main

    model = _tiny_model()
    monkeypatch.setattr(app_main.state, "llm", model)
    monkeypatch.setattr(app_main.state, "tokenizer", _FakeTokenizer())
    monkeypatch.setattr(app_main.state, "device", torch.device("cpu"))

    try:
        app_main.understand(app_main.UnderstandRequest(text="   "))
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400

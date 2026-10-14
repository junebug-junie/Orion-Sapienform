#!/usr/bin/env python3
"""Extractor bake-off: GLiNER variants vs the BERT/RoBERTa/spaCy NER baselines.

Periodic eval, not a CI gate: needs torch/transformers/gliner and live, READ-ONLY
access to ``chat_history_log`` (the labeled texts are Juniper's chat prompts; the
committed label file stores only row ids, a text hash, and span offsets).

    python orion/person_model/evals/run_extractor_bakeoff.py \
        --labels orion/person_model/evals/mention_labels.jsonl \
        --dsn postgresql://postgres:postgres@localhost:55432/conjourney \
        --out /tmp/person-bakeoff/results.json

Scoring (spec section 4): person-mention precision/recall/F1 with relaxed
(overlap) one-to-one matching; predictions overlapping an ``ignore`` span (AI
agents and hosts: Orion, Claude, Muse, Lark) are not scored. Each model's
threshold is swept and the best F1 with precision >= 0.95 is reported next to its
best unconstrained F1. Relation-phrase precision/recall for the GLiNER models
(the baselines have no such label). CPU latency per prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.person_model.extract import PERSON_LABEL, RELATION_LABEL, spans_to_mentions  # noqa: E402

GLINER_MODELS = {
    "gliner_medium-v2.1": ("urchade/gliner_medium-v2.1", "40ec419335d09393f298636f471328b722c6da9e"),
    "gliner_large-v2.1": ("urchade/gliner_large-v2.1", "abd49a1f1ebc12af1be84d06f6848221cf96dcad"),
    "gliner_multi_pii-v1": ("urchade/gliner_multi_pii-v1", "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"),
    "gliner-multitask-large-v0.5": ("knowledgator/gliner-multitask-large-v0.5", "7a95e168036db9ec6f914c0cc6b218edbd87f310"),
}
HF_MODELS = {
    "dslim/bert-base-NER": ("dslim/bert-base-NER", "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"),
    "roberta-large-ner-english": ("Jean-Baptiste/roberta-large-ner-english", "8f3abc1ef81ffbbb0e80568d4fed1dd10d459548"),
}
GLINER_LABELS = (PERSON_LABEL, RELATION_LABEL, "place", "organization")
# Label-phrasing ablation: the bare "person" label (what rev 2 of the spec listed).
GLINER_LABELS_BARE = ("person", RELATION_LABEL, "place", "organization")
SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90


def overlaps(a: Sequence[int], b: Sequence[int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def match(preds: List[Tuple[int, int]], gold: List[List[int]]) -> Tuple[int, int, int]:
    used = set()
    tp = 0
    for p in preds:
        for i, g in enumerate(gold):
            if i not in used and overlaps(p, g):
                used.add(i)
                tp += 1
                break
    return tp, len(preds) - tp, len(gold) - tp


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def score_model(items, preds_by_item, kind: str) -> Dict[str, object]:
    """preds_by_item: list of [(start, end, label, score)] per item."""
    gold_key = "persons" if kind == "person" else "relations"
    want = PERSON_LABEL if kind == "person" else RELATION_LABEL
    sweep = []
    for t in SWEEP:
        tp = fp = fn = 0
        for item, preds in zip(items, preds_by_item):
            ps = [
                (s, e)
                for s, e, lab, sc in preds
                if lab == want and sc >= t and not any(overlaps((s, e), ig) for ig in item["ignore"])
            ]
            a, b, c = match(ps, item[gold_key])
            tp, fp, fn = tp + a, fp + b, fn + c
        sweep.append({"threshold": t, **prf(tp, fp, fn)})
    best = max(sweep, key=lambda r: (r["f1"], r["precision"]))
    gated = [r for r in sweep if r["precision"] >= 0.95]
    best_gated = max(gated, key=lambda r: (r["f1"], r["threshold"])) if gated else None
    return {"best_f1": best, "best_f1_at_precision_0.95": best_gated, "sweep": sweep}


def load_items(labels_path: Path, dsn: str) -> List[dict]:
    import psycopg2

    items = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [it["source_id"] for it in items]
    conn = psycopg2.connect(dsn)
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("select id, prompt from chat_history_log where id = any(%s)", (ids,))
            texts = dict(cur.fetchall())
    finally:
        conn.close()
    for it in items:
        text = texts.get(it["source_id"])
        if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != it["text_sha256"]:
            raise SystemExit(f"label text drifted or missing: {it['source_id']}")
        it["text"] = text
    return items


def run_gliner(name: str, model_id: str, rev: str, items: List[dict], labels=GLINER_LABELS):
    import torch
    from gliner import GLiNER

    torch.use_deterministic_algorithms(True)
    model = GLiNER.from_pretrained(model_id, revision=rev, map_location="cpu")
    model.eval()
    out, t0 = [], time.perf_counter()
    with torch.no_grad():
        for it in items:
            ents = model.predict_entities(it["text"], list(labels), threshold=min(SWEEP))
            # The bare "person" label is scored as the person label for the ablation.
            out.append([(e["start"], e["end"], PERSON_LABEL if e["label"] == "person" else e["label"], float(e["score"])) for e in ents])
    ms = (time.perf_counter() - t0) * 1000 / len(items)
    return out, ms, model


def run_hf(model_id: str, rev: str, items: List[dict]):
    from transformers import pipeline

    nlp = pipeline("ner", model=model_id, revision=rev, aggregation_strategy="simple", device=-1)
    out, t0 = [], time.perf_counter()
    for it in items:
        ents = nlp(it["text"])
        out.append(
            [(int(e["start"]), int(e["end"]), PERSON_LABEL if e["entity_group"] == "PER" else e["entity_group"], float(e["score"])) for e in ents]
        )
    return out, (time.perf_counter() - t0) * 1000 / len(items)


def run_spacy(items: List[dict]):
    import spacy

    nlp = spacy.load("en_core_web_sm")
    out, t0 = [], time.perf_counter()
    for it in items:
        doc = nlp(it["text"])
        out.append([(e.start_char, e.end_char, PERSON_LABEL if e.label_ == "PERSON" else e.label_, 1.0) for e in doc.ents])
    return out, (time.perf_counter() - t0) * 1000 / len(items)


def replay_identical(model, items: List[dict], threshold: float) -> bool:
    """Run the pinned model twice; mention rows must be byte-identical."""
    import torch

    def once():
        rows = []
        with torch.no_grad():
            for it in items:
                ents = model.predict_entities(it["text"], list(GLINER_LABELS), threshold=threshold)
                spans = [{"start": e["start"], "end": e["end"], "label": e["label"], "score": e["score"]} for e in ents]
                rows += [m.model_dump_json() for m in spans_to_mentions(spans, text=it["text"], source_kind="chat_prompt", source_id=it["source_id"], version="replay")]
        return rows

    return once() == once()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path(__file__).with_name("mention_labels.jsonl"))
    ap.add_argument("--dsn", default="postgresql://postgres:postgres@localhost:55432/conjourney")
    ap.add_argument("--out", type=Path, default=Path("/tmp/person-bakeoff/results.json"))
    ap.add_argument("--models", default="all")
    args = ap.parse_args()

    items = load_items(args.labels, args.dsn)
    print(f"items={len(items)} gold_persons={sum(len(i['persons']) for i in items)} gold_relations={sum(len(i['relations']) for i in items)}")
    results: Dict[str, dict] = {}
    pick = None if args.models == "all" else set(args.models.split(","))
    models_kept = {}
    for name, (mid, rev) in GLINER_MODELS.items():
        if pick and name not in pick:
            continue
        preds, ms, model = run_gliner(name, mid, rev, items)
        models_kept[name] = model
        results[name] = {"model_id": mid, "revision": rev, "latency_ms_per_prompt_cpu": round(ms, 1),
                         "person": score_model(items, preds, "person"), "relation": score_model(items, preds, "relation")}
        print(name, json.dumps({k: results[name]["person"][k] for k in ("best_f1", "best_f1_at_precision_0.95")}))
        bare, _, _ = run_gliner(name, mid, rev, items, labels=GLINER_LABELS_BARE)
        results[name]["person_bare_label_ablation"] = score_model(items, bare, "person")
        print(name, "bare-label", json.dumps({k: results[name]["person_bare_label_ablation"][k] for k in ("best_f1", "best_f1_at_precision_0.95")}))
    for name, (mid, rev) in HF_MODELS.items():
        if pick and name not in pick:
            continue
        preds, ms = run_hf(mid, rev, items)
        results[name] = {"model_id": mid, "revision": rev, "latency_ms_per_prompt_cpu": round(ms, 1), "person": score_model(items, preds, "person")}
        print(name, json.dumps({k: results[name]["person"][k] for k in ("best_f1", "best_f1_at_precision_0.95")}))
    if not pick or "spacy" in pick:
        preds, ms = run_spacy(items)
        results["spacy_en_core_web_sm"] = {"model_id": "en_core_web_sm", "latency_ms_per_prompt_cpu": round(ms, 1), "person": score_model(items, preds, "person")}
    for name, model in models_kept.items():
        results[name]["replay_byte_identical"] = replay_identical(model, items, 0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

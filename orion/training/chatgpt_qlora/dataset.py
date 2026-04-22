from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import CanonicalSftExample, DatasetBuildConfig, DatasetManifest, ensure_dir


def _stable_bucket(key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket


def _chat_template(prompt: str, response: str) -> str:
    return f"<|user|>\n{prompt.strip()}\n<|assistant|>\n{response.strip()}"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_from_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def _load_from_postgres(uri: str, import_run_ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    try:
        import psycopg  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("psycopg is required for postgres substrate reads") from exc

    where_clauses = []
    params: list[Any] = []
    if import_run_ids:
        where_clauses.append("import_run_id = ANY(%s)")
        params.append(import_run_ids)

    query = (
        "SELECT example_id, import_run_id, conversation_id, session_id, user_message_id, assistant_message_id, "
        "turn_id, prompt, response, tags, metadata "
        "FROM chat_gpt_derived_example"
    )
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY import_run_id, conversation_id, example_id"
    if limit:
        query += " LIMIT %s"
        params.append(limit)

    with psycopg.connect(uri) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, params)
            return list(cur.fetchall())


def _normalize_record(raw: dict[str, Any], cfg: DatasetBuildConfig) -> CanonicalSftExample | None:
    prompt = str(raw.get("prompt") or "").strip()
    response = str(raw.get("response") or "").strip()
    if len(prompt) < cfg.min_prompt_chars or len(response) < cfg.min_response_chars:
        return None

    example_id = str(raw.get("example_id") or "").strip()
    import_run_id = str(raw.get("import_run_id") or "").strip()
    conversation_id = str(raw.get("conversation_id") or "").strip()
    if not example_id or not import_run_id or not conversation_id:
        return None

    split = "val" if _stable_bucket(example_id, cfg.split_seed) < cfg.val_ratio else "train"
    tags = raw.get("tags")
    if not isinstance(tags, list):
        tags = []
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return CanonicalSftExample(
        example_id=example_id,
        split=split,
        prompt=prompt,
        response=response,
        text=_chat_template(prompt, response),
        import_run_id=import_run_id,
        conversation_id=conversation_id,
        turn_id=raw.get("turn_id"),
        user_message_id=raw.get("user_message_id"),
        assistant_message_id=raw.get("assistant_message_id"),
        tags=[str(t) for t in tags],
        metadata=metadata,
    )


def build_sft_dataset(cfg: DatasetBuildConfig) -> DatasetManifest:
    out_dir = ensure_dir(cfg.output_dir) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.foundry_build_dir:
        partition_file = Path(cfg.foundry_build_dir) / f"partition.{cfg.foundry_partition}.jsonl"
        raw_rows = _load_from_jsonl(partition_file, cfg.source.limit)
    elif cfg.source.examples_jsonl:
        raw_rows = _load_from_jsonl(Path(cfg.source.examples_jsonl), cfg.source.limit)
    else:
        assert cfg.source.postgres_uri
        raw_rows = _load_from_postgres(cfg.source.postgres_uri, cfg.source.import_run_ids, cfg.source.limit)

    included: list[CanonicalSftExample] = []
    excluded = 0
    for row in raw_rows:
        item = _normalize_record(row, cfg)
        if item is None:
            excluded += 1
            continue
        included.append(item)

    all_path = out_dir / "examples.all.jsonl"
    train_path = out_dir / "examples.train.jsonl"
    val_path = out_dir / "examples.val.jsonl"

    with all_path.open("w", encoding="utf-8") as all_fh, train_path.open("w", encoding="utf-8") as train_fh, val_path.open("w", encoding="utf-8") as val_fh:
        for ex in included:
            payload = ex.model_dump(mode="json")
            line = json.dumps(payload, sort_keys=True)
            all_fh.write(line + "\n")
            if ex.split == "train":
                train_fh.write(line + "\n")
            else:
                val_fh.write(line + "\n")

    manifest = DatasetManifest(
        created_at=DatasetManifest.now(),
        source=cfg.source,
        split_seed=cfg.split_seed,
        val_ratio=cfg.val_ratio,
        template_style=cfg.template_style,
        included_count=len(included),
        excluded_count=excluded,
        train_count=sum(1 for x in included if x.split == "train"),
        val_count=sum(1 for x in included if x.split == "val"),
        foundry_build_dir=cfg.foundry_build_dir,
        foundry_partition=cfg.foundry_partition if cfg.foundry_build_dir else None,
        import_run_ids=sorted({x.import_run_id for x in included}),
        conversation_ids=sorted({x.conversation_id for x in included}),
        files={
            "all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
        },
    )
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest

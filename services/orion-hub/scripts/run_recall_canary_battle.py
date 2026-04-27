#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


REQUIRED_CASE_KEYS = {
    "id",
    "query",
    "expected_project_or_domain",
    "expected_anchor_terms",
    "expected_time_sensitivity",
    "notes",
}


@dataclass
class BattleSummary:
    total_cases: int
    successful_runs: int
    failed_runs: int
    selected_profile_id: str
    production_recall_mode: str
    recall_live_apply_enabled: bool
    missing_profile_issues: int
    output_path: str | None
    judgment_counts: dict[str, Any]
    failure_mode_counts: dict[str, Any]


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0, operator_token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.operator_token = operator_token

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, method=method, data=body)
        req.add_header("Content-Type", "application/json")
        if self.operator_token:
            req.add_header("X-Orion-Operator-Token", self.operator_token)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"detail": raw}
            raise RuntimeError(f"HTTP {exc.code} {path}: {json.dumps(parsed, ensure_ascii=True)}") from exc

    def get_status(self) -> dict[str, Any]:
        return self._request_json("GET", "/api/substrate/recall-canary/status")

    def post_query(self, *, query_text: str, profile_id: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/substrate/recall-canary/query",
            payload={"query_text": query_text, "profile_id": profile_id},
        )


def load_battle_fixture(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("fixture must be a JSON array")
    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"fixture row {idx} must be an object")
        missing = sorted(REQUIRED_CASE_KEYS - set(row.keys()))
        if missing:
            raise ValueError(f"fixture row {idx} missing keys: {', '.join(missing)}")
        cases.append(dict(row))
    return cases


def resolve_profile(status_payload: dict[str, Any], requested_profile_id: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    data = dict(status_payload.get("data") or {})
    profiles = list(data.get("available_profiles") or [])
    by_id = {str(item.get("profile_id")): dict(item) for item in profiles if item.get("profile_id")}
    default_id = str(data.get("default_canary_profile_id") or "").strip()
    chosen = (requested_profile_id or "").strip() or default_id
    if not chosen:
        raise ValueError("No profile_id supplied and no default_canary_profile_id available.")
    if chosen not in by_id:
        allowed = ", ".join(sorted(by_id.keys())) or "<none>"
        raise ValueError(f"Invalid profile_id '{chosen}'. Allowed profile_ids: {allowed}")
    return chosen, by_id[chosen], data


def _fmt_row(cols: list[str], widths: list[int]) -> str:
    return " | ".join(col[:width].ljust(width) for col, width in zip(cols, widths))


def print_case_table(rows: list[dict[str, Any]]) -> None:
    headers = ["case_id", "query", "canary_run_id", "selected_profile", "production_mode", "status"]
    widths = [18, 42, 40, 30, 16, 18]
    print(_fmt_row(headers, widths))
    print("-" * (sum(widths) + (3 * (len(widths) - 1))))
    for row in rows:
        print(
            _fmt_row(
                [
                    str(row.get("case_id") or ""),
                    str(row.get("query") or ""),
                    str(row.get("canary_run_id") or ""),
                    str(row.get("selected_profile_id") or ""),
                    str(row.get("production_recall_mode") or ""),
                    str(row.get("status") or ""),
                ],
                widths,
            )
        )


def run_battle(
    *,
    client: ApiClient,
    fixture_cases: list[dict[str, Any]],
    requested_profile_id: str | None,
    output_path: Path | None,
) -> BattleSummary:
    status_payload = client.get_status()
    selected_profile_id, selected_profile, status_data = resolve_profile(status_payload, requested_profile_id)
    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    output_handle = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    try:
        for case in fixture_cases:
            case_id = str(case.get("id") or "")
            query_text = str(case.get("query") or "")
            row: dict[str, Any] = {
                "case_id": case_id,
                "query": query_text,
                "selected_profile_id": selected_profile_id,
            }
            try:
                payload = client.post_query(query_text=query_text, profile_id=selected_profile_id)
                data = dict(payload.get("data") or {})
                row.update(
                    {
                        "canary_run_id": data.get("canary_run_id"),
                        "production_recall_mode": data.get("production_recall_mode"),
                        "status": "ok",
                        "compare_summary": data.get("comparison") or {},
                    }
                )
                successful += 1
                if output_handle is not None:
                    output_handle.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "case": case,
                                "request": {"query_text": query_text, "profile_id": selected_profile_id},
                                "response": data,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            except Exception as exc:
                failed += 1
                row.update(
                    {
                        "canary_run_id": None,
                        "production_recall_mode": status_data.get("production_recall_mode") or "v1",
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                if output_handle is not None:
                    output_handle.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "case": case,
                                "request": {"query_text": query_text, "profile_id": selected_profile_id},
                                "error": str(exc),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            results.append(row)
    finally:
        if output_handle is not None:
            output_handle.close()

    print("")
    print(f"selected_profile_id={selected_profile_id}")
    print(
        f"selected_profile_label={selected_profile.get('label')} "
        f"selected_profile_status={selected_profile.get('status')}"
    )
    print_case_table(results)

    summary = BattleSummary(
        total_cases=len(fixture_cases),
        successful_runs=successful,
        failed_runs=failed,
        selected_profile_id=selected_profile_id,
        production_recall_mode=str(status_data.get("production_recall_mode") or "v1"),
        recall_live_apply_enabled=bool(status_data.get("recall_live_apply_enabled") is True),
        missing_profile_issues=0,
        output_path=str(output_path) if output_path else None,
        judgment_counts=dict(status_data.get("judgment_counts") or {}),
        failure_mode_counts=dict(status_data.get("failure_mode_counts") or {}),
    )
    print("")
    print("battle_rollup")
    print(f"  total_cases={summary.total_cases}")
    print(f"  successful_canary_runs={summary.successful_runs}")
    print(f"  failed_canary_runs={summary.failed_runs}")
    print(f"  selected_profile_id={summary.selected_profile_id}")
    print(f"  production_recall_mode={summary.production_recall_mode}")
    print(f"  recall_live_apply_enabled={summary.recall_live_apply_enabled}")
    print(f"  missing_or_invalid_profile_issues={summary.missing_profile_issues}")
    print(f"  output_file={summary.output_path or '--'}")
    print(f"  judgment_counts={json.dumps(summary.judgment_counts, ensure_ascii=True)}")
    print(f"  failure_mode_counts={json.dumps(summary.failure_mode_counts, ensure_ascii=True)}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Recall Canary battle fixture against selected canary profile.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Hub base URL.")
    parser.add_argument("--profile-id", default=None, help="Recall canary profile_id. Defaults to status.default_canary_profile_id.")
    parser.add_argument("--fixture", required=True, help="Path to battle fixture JSON file.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--operator-token",
        default=None,
        help="Optional operator token. If omitted, script checks SUBSTRATE_MUTATION_OPERATOR_TOKEN env and then services/orion-hub/.env.",
    )
    return parser


def _load_operator_token(explicit_token: str | None) -> str | None:
    token = str(explicit_token or "").strip()
    if token:
        return token
    env_token = str(os.getenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "")).strip()
    if env_token:
        return env_token
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == "SUBSTRATE_MUTATION_OPERATOR_TOKEN":
                candidate = v.strip().strip("'").strip('"')
                if candidate:
                    return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        raise SystemExit(f"fixture not found: {fixture_path}")
    fixture_cases = load_battle_fixture(fixture_path)
    operator_token = _load_operator_token(args.operator_token)
    client = ApiClient(base_url=args.base_url, timeout=float(args.timeout_sec), operator_token=operator_token)
    output_path = Path(args.output) if args.output else None
    run_battle(
        client=client,
        fixture_cases=fixture_cases,
        requested_profile_id=args.profile_id,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

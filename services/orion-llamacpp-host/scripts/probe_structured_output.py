#!/usr/bin/env python3
"""Probe live llama.cpp OpenAI-compatible server for structured-output support."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

PROBE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "name": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["ok", "name", "count"],
    "additionalProperties": False,
}

PROMPT = (
    "Return exactly this data as JSON:\n"
    "ok = true\n"
    'name = "Juniper"\n'
    "count = 3\n"
    "Do not include prose."
)

ADVERSARIAL_PROMPT = (
    'Return JSON with ok=true, name="Juniper", count=3, and also include extra="SHOULD_NOT_APPEAR".'
)

METHOD_ORDER = [
    "json_object_only",
    "json_object_schema",
    "json_schema_schema",
    "json_schema_nested_schema",
    "openai_json_schema_wrapper",
    "no_response_format_control",
]


def _default_base_url() -> str:
    return (
        os.getenv("ATLAS_LLAMACPP_QUICK_URL", "").strip()
        or os.getenv("LLAMACPP_BASE_URL", "").strip()
        or "http://127.0.0.1:8013"
    ).rstrip("/")


def _default_model() -> str:
    return (
        os.getenv("ATLAS_QUICK_CHAT_MODEL", "").strip()
        or os.getenv("LLAMACPP_MODEL", "").strip()
        or "Active-GGUF-Model"
    )


def build_response_format_for_method(method: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if method == "json_object_only":
        return {"type": "json_object"}
    if method == "json_object_schema":
        return {"type": "json_object", "schema": schema}
    if method == "json_schema_schema":
        return {"type": "json_schema", "schema": schema}
    if method == "json_schema_nested_schema":
        return {"type": "json_schema", "json_schema": {"schema": schema}}
    if method == "openai_json_schema_wrapper":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ProbeSchema",
                "strict": True,
                "schema": schema,
            },
        }
    if method == "no_response_format_control":
        return None
    return None


def extract_message_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    if not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return str(content or "")


def _strip_code_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def parse_json_content(text: str) -> Tuple[Optional[Any], Optional[str]]:
    cleaned = _strip_code_fence(text)
    if not cleaned:
        return None, "empty_content"
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"


def validate_probe_schema(obj: Any) -> Tuple[bool, bool, List[str]]:
    """Return (valid_json, schema_valid, issues)."""
    if not isinstance(obj, dict):
        return True, False, ["not_object"]
    issues: List[str] = []
    if obj.get("ok") is not True:
        issues.append("ok_not_true")
    if obj.get("name") != "Juniper":
        issues.append("name_mismatch")
    if obj.get("count") != 3:
        issues.append("count_mismatch")
    allowed = {"ok", "name", "count"}
    extra_keys = [k for k in obj if k not in allowed]
    if extra_keys:
        issues.append(f"extra_keys:{','.join(extra_keys)}")
    schema_valid = not issues
    return True, schema_valid, issues


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def has_forbidden_content(text: str) -> bool:
    raw = text or ""
    if _THINK_OPEN in raw or _THINK_CLOSE in raw:
        return True
    return False


def select_best_method(method_results: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """Prefer first method with schema_enforcement=true."""
    for row in method_results:
        if row.get("name") == "no_response_format_control":
            continue
        if row.get("schema_enforcement") is True:
            return str(row["name"]), True
    for row in method_results:
        if row.get("name") == "json_object_only" and row.get("valid_json"):
            return "json_object_only", False
    return "none", False


def _chat_payload(
    model: str,
    prompt: str,
    response_format: Optional[Dict[str, Any]],
    *,
    include_thinking_kwargs: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0,
        "max_tokens": 120,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if include_thinking_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _post_chat(
    client: Any,
    base_url: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> Tuple[int, Dict[str, Any], Optional[str]]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    try:
        resp = client.post(url, json=payload, timeout=timeout_sec)
    except Exception as exc:  # noqa: BLE001
        return 0, {}, str(exc)
    try:
        body = resp.json() if resp.content else {}
    except Exception:  # noqa: BLE001
        body = {"raw_text": resp.text[:2000]}
    if not isinstance(body, dict):
        body = {"parsed": body}
    if resp.status_code >= 400:
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        return resp.status_code, body, str(err)
    return resp.status_code, body, None


def _evaluate_content(content: str, *, adversarial: bool) -> Dict[str, Any]:
    obj, parse_err = parse_json_content(content)
    valid_json = obj is not None
    schema_valid = False
    schema_enforcement: Optional[bool] = None
    issues: List[str] = []
    if parse_err:
        issues.append(parse_err)
    elif valid_json:
        _, schema_valid, issues = validate_probe_schema(obj)
        if adversarial and isinstance(obj, dict) and "extra" in obj:
            schema_enforcement = False
        elif adversarial and schema_valid:
            schema_enforcement = True
    prose_outside = bool(content.strip()) and not valid_json
    if valid_json and content.strip() and not content.strip().startswith(("{", "[")):
        prose_outside = True
    return {
        "valid_json": valid_json,
        "schema_valid": schema_valid,
        "schema_enforcement": schema_enforcement,
        "issues": issues,
        "forbidden_thinking": has_forbidden_content(content),
        "prose_outside_json": prose_outside,
    }


def probe_method(
    client: Any,
    *,
    base_url: str,
    model: str,
    method: str,
    schema: Dict[str, Any],
    timeout_sec: float,
    artifact_dir: Path,
    include_thinking_on_probe: bool,
    verbose: bool,
) -> Dict[str, Any]:
    response_format = build_response_format_for_method(method, schema)
    thinking_supported = True
    row: Dict[str, Any] = {
        "name": method,
        "http_ok": False,
        "status_code": 0,
        "valid_json": False,
        "schema_valid": False,
        "schema_enforcement": None,
        "content_preview": "",
        "error": None,
        "thinking_control_supported": True,
        "adversarial_schema_enforcement": None,
    }

    def run_once(prompt: str, label: str, adversarial: bool) -> Dict[str, Any]:
        nonlocal thinking_supported
        # Production artifact path always sends enable_thinking=false; baseline omits kwargs.
        use_ctk = method != "no_response_format_control"
        if include_thinking_on_probe and method != "no_response_format_control":
            use_ctk = True
        payload = _chat_payload(
            model,
            prompt,
            response_format,
            include_thinking_kwargs=use_ctk,
        )
        status, body, err = _post_chat(client, base_url, payload, timeout_sec)
        retried_without_ctk = False
        if status >= 400 and use_ctk and payload.get("chat_template_kwargs"):
            err_s = str(err or body).lower()
            if "chat_template" in err_s or "template_kwargs" in err_s or status in (400, 422):
                thinking_supported = False
                retried_without_ctk = True
                payload = _chat_payload(model, prompt, response_format, include_thinking_kwargs=False)
                status, body, err = _post_chat(client, base_url, payload, timeout_sec)
        content = extract_message_content(body) if status == 200 else ""
        eval_row = _evaluate_content(content, adversarial=adversarial)
        artifact = {
            "method": method,
            "label": label,
            "request": payload,
            "status_code": status,
            "response": body,
            "content": content,
            "evaluation": eval_row,
            "retried_without_chat_template_kwargs": retried_without_ctk,
        }
        (artifact_dir / f"{method}_{label}.json").write_text(
            json.dumps(artifact, indent=2, default=str),
            encoding="utf-8",
        )
        if verbose:
            print(f"  [{method}/{label}] status={status} valid_json={eval_row['valid_json']}", file=sys.stderr)
        return {
            "status": status,
            "body": body,
            "err": err,
            "content": content,
            "eval": eval_row,
        }

    primary = run_once(PROMPT, "primary", adversarial=False)
    row["status_code"] = primary["status"]
    row["http_ok"] = primary["status"] == 200
    row["error"] = primary["err"]
    row["thinking_control_supported"] = thinking_supported
    if primary["status"] == 200:
        ev = primary["eval"]
        row["valid_json"] = ev["valid_json"]
        row["schema_valid"] = ev["schema_valid"]
        row["content_preview"] = (primary["content"] or "")[:240]
        if method == "json_object_only" and ev["valid_json"]:
            row["schema_enforcement"] = False
        elif method == "no_response_format_control":
            row["schema_enforcement"] = False
        elif method in METHOD_ORDER and method not in (
            "json_object_only",
            "no_response_format_control",
        ):
            adv = run_once(ADVERSARIAL_PROMPT, "adversarial", adversarial=True)
            if adv["status"] == 200:
                row["adversarial_schema_enforcement"] = adv["eval"].get("schema_enforcement")
                row["schema_enforcement"] = adv["eval"].get("schema_enforcement")
            else:
                row["schema_enforcement"] = False
    return row


def run_probe(
    *,
    base_url: str,
    model: str,
    timeout_sec: float,
    artifact_dir: Path,
    include_thinking_on_probe: bool,
    verbose: bool,
    client_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    if httpx is None and client_factory is None:
        raise RuntimeError("httpx is required for probe_structured_output")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    health_ok = False
    extras: Dict[str, Any] = {}

    client = client_factory() if client_factory else httpx.Client()
    own_client = client_factory is None
    try:
        try:
            hr = client.get(f"{base_url.rstrip('/')}/health", timeout=min(10.0, timeout_sec))
            health_ok = hr.status_code == 200
            extras["health_status"] = hr.status_code
        except Exception as exc:  # noqa: BLE001
            extras["health_error"] = str(exc)

        for path in ("/props", "/slots", "/models"):
            try:
                r = client.get(f"{base_url.rstrip('/')}{path}", timeout=5.0)
                if r.status_code == 200:
                    extras[path.strip("/")] = r.json() if r.content else {}
            except Exception:
                pass

        methods_out: List[Dict[str, Any]] = []
        for name in METHOD_ORDER:
            methods_out.append(
                probe_method(
                    client,
                    base_url=base_url,
                    model=model,
                    method=name,
                    schema=PROBE_SCHEMA,
                    timeout_sec=timeout_sec,
                    artifact_dir=artifact_dir,
                    include_thinking_on_probe=include_thinking_on_probe,
                    verbose=verbose,
                )
            )
            time.sleep(0.05)

        best_method, recommended = select_best_method(methods_out)
        thinking_flags = [m.get("thinking_control_supported") for m in methods_out]
        thinking_control_supported = all(v is not False for v in thinking_flags if v is not None)

        summary = {
            "base_url": base_url,
            "model": model,
            "health_ok": health_ok,
            "thinking_control_supported": thinking_control_supported,
            "methods": methods_out,
            "best_method": best_method,
            "recommended_for_memory_graph": recommended,
            "probe_schema": PROBE_SCHEMA,
            "extras": extras,
            "artifact_dir": str(artifact_dir),
        }
        (artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        if own_client:
            client.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Probe llama.cpp structured output methods")
    parser.add_argument("--base-url", default=_default_base_url())
    parser.add_argument("--model", default=_default_model())
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--artifact-dir",
        default="services/orion-llamacpp-host/artifacts/structured-output-probe/latest",
    )
    parser.add_argument(
        "--include-thinking-on-probe",
        action="store_true",
        help="Send chat_template_kwargs.enable_thinking=false (default: off unless flag set)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    summary = run_probe(
        base_url=str(args.base_url).rstrip("/"),
        model=str(args.model),
        timeout_sec=float(args.timeout_sec),
        artifact_dir=Path(args.artifact_dir),
        include_thinking_on_probe=bool(args.include_thinking_on_probe),
        verbose=bool(args.verbose),
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("health_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the episode distiller offline on real past episodes (spec Stage 1 acceptance 2-9).

    python services/orion-memory-consolidation/evals/run_episode_distill_eval.py \
        --route agent --compare-route quick_background --repeat 2 --out /tmp/memory-distill-eval

* Episodes: the Austin morning (2026-09-28 06:26-09:54 UTC, the spec's worked example) plus the five
  most recent other Rule 3 episodes with at least 3 content turns, from the 30-day boundary replay
  fixture. Turn text is read READ-ONLY from Postgres (full, untruncated) at run time.
* The model is called through the live LLM gateway over the bus (route ``--route``; the new
  ``memory_distill`` route only exists once this branch deploys, so the eval uses ``agent``, the
  same 27B class). ``--compare-route`` runs the 8B on the same episodes (label-free comparison).
* Everything private (statements, quotes) goes ONLY to ``--out`` (default under /tmp). What this
  prints and what ``--summary-json`` writes are counts, rates, tokens and latency: safe to commit.

Checks reported (label-free; nobody labels anything):
  grounding (juniper_said with a verified prompt quote), downgrades, rejections by reason,
  coverage of non-command turns, junk (<=5 words / duplicates /
  command-only), referent-set Jaccard between repeated runs, tokens and latency per episode, and
  stakes as stakes:category per kept memory (``--live`` adds the episodes the deployed shadow
  distiller already ran, with the stakes their stored memories got as the "before"), and
  the Austin property checks (event referent with alias "austin"; juniper_said memory with a
  verified "introvert" quote; a follow_up due on/after 2026-09-30; no memory from command turns).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from orion.memory.episode.distill import parse_distillation, render_prompt, turns_from_rows  # noqa: E402
from orion.memory.episode.validate import coverage, validate_distillation  # noqa: E402

PG_CONTAINER = "orion-athena-sql-db"
AUSTIN = ("2026-09-28T06:00:00+00:00", "2026-09-28T10:30:00+00:00")
DEFAULT_BUS = os.environ.get("ORION_BUS_URL", "redis://100.92.216.81:6379/0")


def _psql_json(sql: str) -> Any:
    out = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "postgres", "-d", "conjourney", "-v", "ON_ERROR_STOP=1",
         "-Atc", f"BEGIN READ ONLY; {sql}; ROLLBACK;"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and ln not in {"BEGIN", "ROLLBACK"}]
    return json.loads(lines[-1])


def select_episodes(n_other: int = 5) -> list[dict[str, Any]]:
    import run_episode_boundary_replay_eval as replay

    fixture = json.loads(replay.FIXTURE.read_text(encoding="utf-8"))
    turns = replay.annotate(fixture["turns"])
    a0, a1 = (datetime.fromisoformat(x) for x in AUSTIN)
    austin = [t for t in turns if a0 <= t["at_dt"] <= a1]
    episodes = [{"name": "austin", "turn_ids": [t["correlation_id"] for t in austin]}]
    others = []
    for ep in replay.split_v2(turns, score_key="first_pass"):
        ids = [t["correlation_id"] for t in ep["turns"]]
        content = [t for t in ep["turns"] if not t.get("is_command")]
        if len(content) >= 3 and not any(a0 <= t["at_dt"] <= a1 for t in ep["turns"]):
            others.append({"name": f"ep-{ep['turns'][0]['at_dt']:%m%d-%H%M}", "turn_ids": ids})
    episodes += others[-n_other:]
    return episodes


def live_distilled_episodes() -> list[dict[str, Any]]:
    """Episodes the deployed shadow distiller already ran (episode_distill_run), with the stakes the
    stored memories got (counts only, no text) as the "before"."""
    sql = """
    SELECT coalesce(json_agg(json_build_object(
        'name', 'live-' || left(r.episode_id::text, 8),
        'episode_id', r.episode_id::text,
        'prompt_version', r.prompt_version,
        'turn_ids', (SELECT json_agg(t->>'correlation_id') FROM jsonb_array_elements(s.turns) t),
        'before_stakes', (SELECT json_object_agg(k, n) FROM (
            SELECT m.stakes || ':' || coalesce(m.stakes_reason, '-') AS k, count(*) AS n
            FROM episode_memory m WHERE m.episode_id::text = r.episode_id::text GROUP BY 1) x)
      ) ORDER BY s.started_at), '[]'::json)
    FROM episode_distill_run r JOIN memory_episode_shadow s ON s.episode_id::text = r.episode_id::text
    """
    return _psql_json(sql)


def load_rows(turn_ids: list[str]) -> list[dict[str, Any]]:
    ids = ",".join("'" + i.replace("'", "") + "'" for i in turn_ids)
    sql = (f"SELECT coalesce(json_agg(json_build_object('correlation_id', correlation_id, 'prompt', prompt, "
           f"'response', response, 'created_at', created_at) ORDER BY created_at), '[]'::json) "
           f"FROM chat_history_log WHERE correlation_id IN ({ids})")
    return _psql_json(sql)


async def call_gateway(bus, prompt: str, *, route: str, timeout: float, max_tokens: int) -> dict[str, Any]:
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)], route=route,
        options={"llm_route": route, "max_tokens": max_tokens, "temperature": 0.2,
                 "purpose": "memory_episode_distill_eval", "structured_output_method": "json_object_only",
                 "chat_template_kwargs": {"enable_thinking": False}, "skip_spark_candidate_publish": True,
                 "gateway_read_timeout_sec": timeout})
    corr = uuid4()
    reply = f"orion:exec:result:LLMGatewayService:{corr}"
    env = BaseEnvelope(kind="llm.chat.request", correlation_id=corr, reply_to=reply,
                       source=ServiceRef(name="memory-distill-eval", version="0", node="athena"),
                       payload=payload.model_dump(mode="json"))
    started = time.monotonic()
    msg = await bus.rpc_request("orion:exec:request:LLMGatewayService", env, reply_channel=reply,
                                timeout_sec=timeout + 30)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    result = decoded.envelope.payload or {}
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    if raw.get("error"):
        raise RuntimeError(f"gateway_error:{raw.get('error')}:{raw.get('details')}")
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return {"text": str(result.get("content") or result.get("text") or ""), "usage": usage,
            "model": raw.get("model"), "latency_ms": int((time.monotonic() - started) * 1000)}


def austin_checks(result, turns) -> dict[str, bool]:
    kept = result.memories
    commands = {t.correlation_id for t in turns if t.is_command}
    return {
        "event_referent_with_alias_austin": any(
            k.startswith("event:") and "austin" in k for m in kept for k, _ in m.referents),
        "juniper_said_verified_introvert_quote": any(
            m.voice == "juniper_said" and any(e.verified and "introvert" in e.quote.lower() for e in m.evidence)
            for m in kept),
        "follow_up_due_on_or_after_2026_09_30": any(
            m.purpose == "follow_up" and m.due_after is not None
            and m.due_after >= datetime(2026, 9, 30, tzinfo=timezone.utc) for m in kept),
        "no_kept_memory_rests_only_on_command_turns": not any(
            {e.source_id for e in m.evidence if e.verified} <= commands for m in kept),
    }


def score(result, turns, answer, parsed_count: int) -> dict[str, Any]:
    kept = result.memories
    js = [m for m in kept if m.voice == "juniper_said"]
    aj = [m for m in kept if m.purpose == "about_juniper"]
    return {
        "proposed": parsed_count,
        "kept": len(kept),
        "rejections": dict(Counter(r.reason for r in result.rejections)),
        "downgrades": result.downgrades,
        "juniper_said": len(js),
        "juniper_said_grounded": sum(1 for m in js if any(e.verified and e.source_kind == "chat_prompt" for e in m.evidence)),
        "about_juniper": len(aj),
        "high_stakes": sum(1 for m in kept if m.stakes == "high"),
        # Stakes after validation, as stakes:category (no text). "high:-" = high without a category.
        "stakes": dict(Counter(f"{m.stakes}:{m.stakes_reason or '-'}" for m in kept)),
        # As the distiller proposed them, before the consistency check resolved anything.
        "stakes_events": dict(Counter(e.op for m in kept for e in m.events if e.op.startswith("stakes"))),
        # The renderer contract is first person; a statement naming Orion is written about Orion, not by it.
        "statements_naming_orion": sum(1 for m in kept if "orion" in m.statement.lower().split()
                                       or "orion's" in m.statement.lower()),
        "questions": len(result.questions),
        "voices": dict(Counter(m.voice for m in kept)),
        "purposes": dict(Counter(m.purpose for m in kept)),
        **coverage(result, turns),
        "prompt_tokens": (answer.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (answer.get("usage") or {}).get("completion_tokens"),
        "latency_ms": answer.get("latency_ms"),
        "model": answer.get("model"),
    }


def _private_dump(result) -> dict[str, Any]:
    return {
        "memories": [{"purpose": m.purpose, "voice": m.voice, "channel": m.channel, "statement": m.statement,
                      "stakes": m.stakes, "stakes_reason": m.stakes_reason, "referents": m.referents,
                      "evidence": [vars(e) for e in m.evidence], "events": [vars(e) for e in m.events],
                      "due_after": m.due_after.isoformat() if m.due_after else None} for m in result.memories],
        "questions": [{"text": q.text, "scope": q.scope} for q in result.questions],
        "rejections": [{"reason": r.reason, "candidate": r.candidate} for r in result.rejections],
    }


async def main_async(args) -> dict[str, Any]:
    from orion.core.bus.async_service import OrionBusAsync

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bus = OrionBusAsync(url=args.bus_url)
    await bus.connect()
    summary: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "routes": {}, "episodes": []}
    try:
        episodes = [] if args.live_only else select_episodes()
        if args.live or args.live_only:
            episodes += live_distilled_episodes()
        episodes = [e for e in episodes if not args.only or e["name"] in args.only]
        for ep in episodes:
            rows = load_rows(ep["turn_ids"])
            turns = turns_from_rows(rows)
            prompt = render_prompt(episode_id=ep["name"], turns=turns)
            info = {"name": ep["name"], "before_stakes": ep.get("before_stakes"),
                    "before_prompt_version": ep.get("prompt_version"), "turns": len(turns), "commands": sum(t.is_command for t in turns),
                    "chars": sum(len(t.prompt) + len(t.response) for t in turns), "runs": {}}
            for route in [args.route] + ([args.compare_route] if args.compare_route else []):
                runs = []
                for i in range(args.repeat if route == args.route else 1):
                    try:
                        answer = await call_gateway(bus, prompt, route=route, timeout=args.timeout,
                                                    max_tokens=args.max_tokens)
                        # Save the raw answer first: a scoring bug must never cost a model call.
                        (out / f"{ep['name']}.{route}.{i}.answer.txt").write_text(answer["text"])
                        parsed = parse_distillation(answer["text"])
                        result = validate_distillation(parsed, turns, episode_id=ep["name"])
                        s = score(result, turns, answer, len(parsed.memories))
                        if ep["name"] == "austin":
                            s["austin_checks"] = austin_checks(result, turns)
                        s["referents"] = sorted({k for m in result.memories for k, _ in m.referents})
                        (out / f"{ep['name']}.{route}.{i}.json").write_text(json.dumps(
                            {"answer": answer["text"], "validated": _private_dump(result)}, indent=1, default=str))
                    except Exception as exc:  # noqa: BLE001
                        s = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                    runs.append(s)
                    print(f"{ep['name']} {route} run{i}: " + json.dumps({k: v for k, v in s.items() if k != 'referents'}),
                          flush=True)
                if len(runs) >= 2 and all("referents" in r for r in runs[:2]):
                    a, b = set(runs[0]["referents"]), set(runs[1]["referents"])
                    info.setdefault("jaccard", {})[route] = round(len(a & b) / len(a | b), 3) if a | b else 1.0
                for r in runs:
                    r.pop("referents", None)
                info["runs"][route] = runs
            summary["episodes"].append(info)
    finally:
        await bus.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="agent")
    ap.add_argument("--compare-route", default="quick_background")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--bus-url", default=DEFAULT_BUS)
    ap.add_argument("--out", default="/tmp/memory-distill-eval")
    ap.add_argument("--summary-json", type=Path)
    ap.add_argument("--only", nargs="*", help="episode names to run (default: all)")
    ap.add_argument("--template", type=Path, help="render this prompt template instead of the current one "
                    "(e.g. the previous version via `git show <rev>:<path>`, for a before/after on the same lane)")
    ap.add_argument("--live", action="store_true", help="also re-run the episodes the deployed distiller ran")
    ap.add_argument("--live-only", action="store_true", help="only the episodes the deployed distiller ran")
    args = ap.parse_args()
    if args.template:
        import orion.memory.episode.distill as distill

        distill.PROMPT_PATH = args.template.resolve()
    summary = asyncio.run(main_async(args))
    text = json.dumps(summary, indent=1, default=str)
    (Path(args.out) / "summary.json").write_text(text)
    if args.summary_json:
        args.summary_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

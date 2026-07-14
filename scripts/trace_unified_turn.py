#!/usr/bin/env python3
"""Unified-turn hop tracer — dump a completed turn or follow one live.

Examples:
  python scripts/trace_unified_turn.py dump 1d7965dc-6cd3-44b2-88af-a001c51fbb45
  python scripts/trace_unified_turn.py latest
  python scripts/trace_unified_turn.py live --corr 1d7965dc-6cd3-44b2-88af-a001c51fbb45
  python scripts/trace_unified_turn.py live --bus
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

# --- Log line parsers (unit-tested) -------------------------------------------------

_CORR_RE = re.compile(
    r"(?:corr(?:elation_id)?|corr_id|trace_id)=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.I,
)

_HOP_PATTERNS: dict[str, re.Pattern[str]] = {
    "ingress_pre_turn": re.compile(r"pre_turn_appraisal_paradigm_result\s+corr=", re.I),
    "stance_react": re.compile(r"stance_react complete\s+corr=", re.I),
    "motor_complete": re.compile(
        r"harness_motor_complete\s+corr=.*steps=(?P<steps>\d+).*grammar_receipts=(?P<grammar>\d+)"
        r".*verdict=(?P<verdict>\S+).*grounding=(?P<grounding>\S+).*draft_len=(?P<draft_len>\d+)",
        re.I,
    ),
    "grammar_step": re.compile(
        r"harness_grammar_step_published\s+corr=.*step=(?P<step>\d+)\s+tool=(?P<tool>\S+)",
        re.I,
    ),
    "substrate_5a": re.compile(
        r"finalize_appraisal complete\s+corr=.*draft_hash=(?P<draft_hash>\S+)\s+surprise=(?P<surprise>[\d.]+)",
        re.I,
    ),
    "cortex_5b": re.compile(
        r"(?:harness cortex RPC -> .*verb=harness_finalize_reflect|"
        r"final_text_assembly\s+corr_id=.*verb=harness_finalize_reflect.*result_len=(?P<result_len>\d+))",
        re.I,
    ),
    "cortex_5c": re.compile(
        r"(?:harness cortex RPC -> .*verb=orion_voice_finalize|"
        r"final_text_assembly\s+corr_id=.*verb=orion_voice_finalize.*result_len=(?P<result_len>\d+))",
        re.I,
    ),
    "verdict": re.compile(
        r"harness_verdict_published\s+corr=.*alignment=(?P<alignment>\S+)",
        re.I,
    ),
    "outcome": re.compile(
        r"harness_turn_outcome_published\s+corr=.*surprise_resolved=(?P<surprise_resolved>\S+)"
        r"(?:\s+finalize_failed=(?P<finalize_failed>\S+))?"
        r".*grammar_events=(?P<grammar_events>\d+)",
        re.I,
    ),
    "harness_complete": re.compile(
        r"harness run complete\s+corr=.*finalize_ran=(?P<finalize_ran>\S+)",
        re.I,
    ),
    "closure_publish": re.compile(
        r"harness_post_turn_closure_(?:published|emitted)\s+corr=.*surprise_unresolved=(?P<surprise_unresolved>\S+)"
        r"(?:.*finalize_failed=(?P<finalize_failed>\S+))?",
        re.I,
    ),
    "system_error": re.compile(
        r"harness_finalize_system_error_published\s+corr=.*channel=(?P<channel>\S+)\s+phase=(?P<phase>\S+)",
        re.I,
    ),
    "closure_received": re.compile(
        r"post_turn_closure received\s+corr=.*surprise_unresolved=(?P<surprise_unresolved>\S+).*outcome_id=(?P<outcome_id>\S+)",
        re.I,
    ),
    "prediction_error": re.compile(
        r"post_turn_closure_prediction_error_write\s+corr=.*node_id=(?P<node_id>\S+)\s+error=(?P<error>[\d.]+)",
        re.I,
    ),
    "context_overflow": re.compile(r"exceed_context_size_error|69774 tokens", re.I),
    "final_text_assembly": re.compile(
        r"final_text_assembly\s+corr_id=.*verb=(?P<verb>\S+).*result_len=(?P<result_len>\d+)",
        re.I,
    ),
}

_DEFAULT_CONTAINERS = (
    "orion-athena-hub",
    "orion-athena-thought",
    "orion-athena-harness-governor",
    "orion-athena-substrate-runtime",
    "orion-athena-cortex-exec",
)

_LIVE_BUS_CHANNELS = (
    "orion:thought:artifact",
    "orion:harness:run:step",
    "orion:grammar:event",
    "orion:harness:verdict:artifact",
    "orion:substrate:turn_outcome",
    "orion:substrate:post_turn_closure",
    "orion:harness:run:artifact",
    "orion:cognition:trace",
)


@dataclass
class HopEvidence:
    hop: str
    service: str
    line: str
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class TurnTrace:
    correlation_id: str
    evidence: list[HopEvidence] = field(default_factory=list)
    grammar_summaries: list[str] = field(default_factory=list)
    hub_trace: dict[str, Any] | None = None

    def hops_seen(self) -> list[str]:
        order = [
            "ingress_pre_turn",
            "stance_react",
            "motor_complete",
            "substrate_5a",
            "cortex_5b",
            "verdict",
            "cortex_5c",
            "outcome",
            "system_error",
            "harness_complete",
            "closure_publish",
            "closure_received",
            "prediction_error",
        ]
        seen = {e.hop for e in self.evidence}
        return [h for h in order if h in seen]


def extract_correlation_id(line: str) -> str | None:
    match = _CORR_RE.search(line)
    return match.group(1).lower() if match else None


def classify_log_line(line: str) -> tuple[str | None, dict[str, str]]:
    for hop, pattern in _HOP_PATTERNS.items():
        match = pattern.search(line)
        if not match:
            continue
        fields = {k: v for k, v in match.groupdict().items() if v is not None}
        return hop, fields
    return None, {}


def parse_service_logs(
    *,
    correlation_id: str,
    lines: Iterable[tuple[str, str]],
) -> list[HopEvidence]:
    corr = correlation_id.lower()
    out: list[HopEvidence] = []
    for service, line in lines:
        if corr not in line.lower():
            continue
        hop, fields = classify_log_line(line)
        if hop is None:
            continue
        out.append(HopEvidence(hop=hop, service=service, line=line.strip(), fields=fields))
    return out


def _resolve_containers(explicit: Sequence[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        names = [n.strip() for n in proc.stdout.splitlines() if n.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return list(_DEFAULT_CONTAINERS)
    picked: list[str] = []
    for default in _DEFAULT_CONTAINERS:
        if default in names:
            picked.append(default)
            continue
        # fuzzy: orion-athena-harness-governor vs orion-*-harness-governor
        token = default.removeprefix("orion-athena-")
        for name in names:
            if token in name:
                picked.append(name)
                break
    return picked or list(_DEFAULT_CONTAINERS)


def _docker_logs(container: str, *, tail: int | None = None) -> list[str]:
    cmd = ["docker", "logs", container]
    if tail is not None and tail > 0:
        cmd[2:2] = ["--tail", str(tail)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return []
    return (proc.stdout + proc.stderr).splitlines()


def _fetch_hub_cognition_trace(correlation_id: str, *, hub_port: int) -> dict[str, Any] | None:
    url = f"http://127.0.0.1:{hub_port}/api/cognition/trace/{correlation_id}"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _fetch_grammar_summaries(correlation_id: str, *, db_url: str | None, limit: int) -> list[str]:
    if not db_url:
        return []
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        return []
    query = text(
        """
        SELECT event_json->'atom'->>'summary' AS summary
        FROM grammar_events
        WHERE trace_id = :corr
        ORDER BY created_at
        LIMIT :lim
        """
    )
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(query, {"corr": correlation_id, "lim": limit}).mappings().all()
        return [str(r["summary"]) for r in rows if r.get("summary")]
    except Exception:
        return []


def _resolve_db_url(cli: str | None) -> str | None:
    if cli:
        return cli
    for key in ("GRAMMAR_ATLAS_POSTGRES_URI", "POSTGRES_URI", "DATABASE_URL", "ORION_SQL_URL"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return None


def _find_latest_correlation_id(containers: Sequence[str], *, tail: int) -> str | None:
    candidates: list[tuple[str, str]] = []
    scan_tail = tail if tail > 0 else 2000
    for container in containers:
        for line in _docker_logs(container, tail=scan_tail):
            if "harness_motor_complete" in line or "harness run complete" in line:
                corr = extract_correlation_id(line)
                if corr:
                    candidates.append((corr, line))
    return candidates[-1][0] if candidates else None


def build_turn_trace(
    correlation_id: str,
    *,
    containers: Sequence[str] | None = None,
    log_tail: int = 0,
    db_url: str | None = None,
    hub_port: int = 8080,
    grammar_limit: int = 8,
) -> TurnTrace:
    resolved = _resolve_containers(containers)
    lines: list[tuple[str, str]] = []
    for container in resolved:
        tail = log_tail if log_tail > 0 else None
        for line in _docker_logs(container, tail=tail):
            lines.append((container, line))
    trace = TurnTrace(
        correlation_id=correlation_id,
        evidence=parse_service_logs(correlation_id=correlation_id, lines=lines),
        grammar_summaries=_fetch_grammar_summaries(correlation_id, db_url=db_url, limit=grammar_limit),
        hub_trace=_fetch_hub_cognition_trace(correlation_id, hub_port=hub_port),
    )
    return trace


def format_turn_trace(trace: TurnTrace, *, verbose: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"# Unified turn trace: {trace.correlation_id}")
    lines.append(f"generated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    if trace.hub_trace:
        lines.append("## Hub cognition trace (cached)")
        lines.append(f"  verb: {trace.hub_trace.get('verb_name')}")
        steps = trace.hub_trace.get("steps") or []
        lines.append(f"  steps: {len(steps)}")
        lines.append("")
    else:
        lines.append("## Hub cognition trace")
        lines.append("  UNVERIFIED — not in hub cache (expired or COGNITION_TRACE_CACHE disabled)")
        lines.append("")

    hop_groups: dict[str, list[HopEvidence]] = defaultdict(list)
    for ev in trace.evidence:
        hop_groups[ev.hop].append(ev)

    hop_labels = {
        "ingress_pre_turn": "1. Ingress / pre-turn repair",
        "stance_react": "3. Stance react (ThoughtEventV1)",
        "motor_complete": "4. Motor (FCC)",
        "grammar_step": "4. Motor grammar steps",
        "substrate_5a": "5a. Substrate finalize appraisal",
        "cortex_5b": "5b. Integrative reflect",
        "verdict": "5b. Verdict molecule",
        "cortex_5c": "5c. Orion voice",
        "outcome": "6b. Outcome molecule",
        "system_error": "Homeostatic system error (finalize failure)",
        "harness_complete": "HarnessRunV1 reply",
        "closure_publish": "7. Post-turn closure (publish)",
        "closure_received": "7. Post-turn closure (substrate)",
        "prediction_error": "7. Prediction error write",
        "context_overflow": "Motor context overflow",
        "final_text_assembly": "Cortex final_text assembly",
    }

    for hop_key in trace.hops_seen():
        label = hop_labels.get(hop_key, hop_key)
        lines.append(f"## {label}")
        for ev in hop_groups.get(hop_key, []):
            field_bits = ", ".join(f"{k}={v}" for k, v in ev.fields.items())
            if field_bits:
                lines.append(f"  [{ev.service}] {field_bits}")
            else:
                lines.append(f"  [{ev.service}] {ev.line[:200]}")
        lines.append("")

    overflow = hop_groups.get("context_overflow", [])
    if overflow:
        lines.append("## Motor context overflow (warning)")
        for ev in overflow[:3]:
            lines.append(f"  [{ev.service}] {ev.line[:200]}")
        lines.append("")

    assembly = hop_groups.get("final_text_assembly", [])
    if assembly:
        lines.append("## Cortex output sizes")
        for ev in assembly:
            lines.append(
                f"  verb={ev.fields.get('verb', '?')} result_len={ev.fields.get('result_len', '?')}"
            )
        lines.append("")

    if trace.grammar_summaries:
        lines.append(f"## Grammar receipts (SQL sample, {len(trace.grammar_summaries)} rows)")
        for summary in trace.grammar_summaries[:6]:
            lines.append(f"  - {summary[:240]}")
        if len(trace.grammar_summaries) > 6:
            lines.append(f"  ... ({len(trace.grammar_summaries) - 6} more in DB)")
        lines.append("")

    missing = [
        label
        for key, label in [
            ("stance_react", "stance"),
            ("motor_complete", "motor"),
            ("substrate_5a", "5a"),
            ("verdict", "5b verdict"),
            ("outcome", "6b outcome"),
            ("closure_received", "7 closure consume"),
        ]
        if key not in hop_groups
    ]
    if missing:
        lines.append("## Missing hops (not found in docker logs)")
        lines.append(f"  {', '.join(missing)}")
        lines.append("")

    if verbose:
        lines.append("## Raw matching lines")
        for ev in trace.evidence:
            lines.append(f"  [{ev.hop}] {ev.service}: {ev.line}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _live_print(hop: str | None, source: str, line: str, *, fields: dict[str, str]) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    if hop:
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[{ts}] {source} :: {hop} {extra}".rstrip(), flush=True)
    else:
        print(f"[{ts}] {source} :: {line[:200]}", flush=True)


async def _live_bus_loop(
    *,
    correlation_id: str | None,
    redis_url: str,
    channels: Sequence[str],
) -> None:
    from orion.core.bus.async_service import OrionBusAsync

    bus = OrionBusAsync(redis_url)
    if not bus.enabled:
        print("Bus disabled; live bus trace unavailable.", file=sys.stderr)
        return
    corr_filter = correlation_id.lower() if correlation_id else None
    print(f"live-bus: subscribing {len(channels)} channels redis={redis_url}", flush=True)
    async with bus.subscribe(*channels) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            mtype = msg.get("type")
            if mtype not in ("message", "pmessage"):
                continue
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok or decoded.envelope is None:
                continue
            env = decoded.envelope
            corr = str(env.correlation_id or "").lower()
            payload_corr = ""
            if isinstance(env.payload, dict):
                payload_corr = str(env.payload.get("correlation_id") or "").lower()
            if corr_filter and corr_filter not in {corr, payload_corr}:
                continue
            print(
                f"[bus] kind={env.kind} corr={corr or payload_corr} channel={msg.get('channel')}",
                flush=True,
            )


def _live_docker_loop(
    *,
    correlation_id: str | None,
    containers: Sequence[str],
    follow: bool,
) -> None:
    corr_filter = correlation_id.lower() if correlation_id else None
    active_corr: str | None = corr_filter
    cmd_base = ["docker", "logs"]
    if follow:
        cmd_base.append("-f")
    cmd_base.append("--tail")
    cmd_base.append("0" if follow else "200")

    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for container in containers:
        try:
            proc = subprocess.Popen(
                [*cmd_base, container],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            print("docker not found", file=sys.stderr)
            return
        procs.append((container, proc))

    print(
        f"live: tailing {len(procs)} containers"
        + (f" corr={active_corr}" if active_corr else " (auto-detect corr on harness start)")
        + " — Ctrl-C to stop",
        flush=True,
    )
    try:
        while True:
            any_alive = False
            for container, proc in procs:
                if proc.stdout is None:
                    continue
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is None:
                        any_alive = True
                    continue
                any_alive = True
                stripped = line.rstrip("\n")
                line_corr = extract_correlation_id(stripped)
                if not active_corr and line_corr and (
                    "orion:thought:request" in stripped
                    or "orion:harness:run:request" in stripped
                    or "pre_turn_appraisal" in stripped
                ):
                    active_corr = line_corr
                    print(f"--- auto-detected correlation_id={active_corr} ---", flush=True)
                if active_corr and active_corr not in stripped.lower():
                    continue
                hop, fields = classify_log_line(stripped)
                if hop or (active_corr and active_corr in stripped.lower()):
                    _live_print(hop, container, stripped, fields=fields)
            if not follow:
                break
            if not any_alive:
                break
    except KeyboardInterrupt:
        print("\nlive: stopped", flush=True)
    finally:
        for _, proc in procs:
            proc.terminate()


def cmd_dump(args: argparse.Namespace) -> int:
    corr = args.correlation_id.lower()
    trace = build_turn_trace(
        corr,
        containers=args.containers,
        log_tail=args.log_tail,
        db_url=_resolve_db_url(args.db_url),
        hub_port=args.hub_port,
        grammar_limit=args.grammar_limit,
    )
    sys.stdout.write(format_turn_trace(trace, verbose=args.verbose))
    return 0


def cmd_latest(args: argparse.Namespace) -> int:
    containers = _resolve_containers(args.containers)
    corr = _find_latest_correlation_id(containers, tail=args.log_tail)
    if not corr:
        print("No recent unified turn found in harness logs.", file=sys.stderr)
        return 1
    if args.dump:
        args.correlation_id = corr
        return cmd_dump(args)
    print(corr)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    containers = _resolve_containers(args.containers)
    corr = args.corr.lower() if args.corr else None
    if getattr(args, "all", False):
        args.bus = True
        args.docker = True
    if args.bus:
        redis_url = args.redis or os.getenv("ORION_BUS_URL", "redis://localhost:6379/0")

        async def _run() -> None:
            bus_task = asyncio.create_task(
                _live_bus_loop(
                    correlation_id=corr,
                    redis_url=redis_url,
                    channels=args.bus_channels or _LIVE_BUS_CHANNELS,
                )
            )
            try:
                await bus_task
            except asyncio.CancelledError:
                pass

        if args.docker:
            def _docker_thread() -> None:
                _live_docker_loop(
                    correlation_id=corr,
                    containers=containers,
                    follow=True,
                )

            thread = threading.Thread(target=_docker_thread, daemon=True)
            thread.start()
            try:
                asyncio.run(
                    _live_bus_loop(
                        correlation_id=corr,
                        redis_url=redis_url,
                        channels=args.bus_channels or _LIVE_BUS_CHANNELS,
                    )
                )
            except KeyboardInterrupt:
                print("\nlive: stopped", flush=True)
        else:
            try:
                asyncio.run(_run())
            except KeyboardInterrupt:
                print("\nlive: stopped", flush=True)
        return 0

    _live_docker_loop(correlation_id=corr, containers=containers, follow=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace unified Orion turns by correlation_id.")
    parser.add_argument("--containers", nargs="*", help="Docker container names (auto-detect default)")
    parser.add_argument(
        "--log-tail",
        type=int,
        default=0,
        help="Docker log lines per container (0 = full logs; use 8000+ if slow)",
    )
    parser.add_argument("--db-url", help="Postgres URL for grammar_events SQL")
    parser.add_argument("--hub-port", type=int, default=int(os.getenv("HUB_PORT", "8080")))
    parser.add_argument("--grammar-limit", type=int, default=12)
    parser.add_argument("-v", "--verbose", action="store_true", help="Include raw log lines in dump")

    sub = parser.add_subparsers(dest="command", required=True)

    p_dump = sub.add_parser("dump", help="One-shot hop report for a correlation_id")
    p_dump.add_argument("correlation_id")
    p_dump.set_defaults(func=cmd_dump)

    p_latest = sub.add_parser("latest", help="Print most recent harness correlation_id")
    p_latest.add_argument("--dump", action="store_true", help="Also run dump on latest corr")
    p_latest.set_defaults(func=cmd_latest)

    p_live = sub.add_parser("live", help="Follow unified-turn hops in real time")
    p_live.add_argument("--corr", help="Filter to correlation_id (omit to auto-detect)")
    p_live.add_argument(
        "--all",
        action="store_true",
        help="Docker logs + bus subscribe (same as --docker --bus)",
    )
    p_live.add_argument(
        "--bus",
        action="store_true",
        help="Subscribe to the bus channels in _LIVE_BUS_CHANNELS instead of docker-log-only mode",
    )
    p_live.add_argument(
        "--docker",
        action="store_true",
        help="With --bus, also tail docker logs (default live mode is docker-only)",
    )
    p_live.add_argument("--redis", help="ORION_BUS_URL for --bus")
    p_live.add_argument("--bus-channels", nargs="*", help="Override bus channels for --bus")
    p_live.set_defaults(func=cmd_live)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

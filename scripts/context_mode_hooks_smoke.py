#!/usr/bin/env python3
"""Headless Context Mode hook-validation smoke for the harness governor.

Stage B of the semantic self-indexing proposal (§9.3): prove — or honestly
fail — each hook-mode requirement before HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED
is turned on for ordinary Orion turns.

Runs INSIDE the harness-governor container (python3 + claude on PATH,
/root/.claude persistent volume with the context-mode plugin installed).
It never modifies repo files and is safe to re-run.

Each check emits one JSON line on stdout:

    {"check": <name>, "status": "PASS"|"FAIL"|"UNVERIFIED", "evidence": ...}

Exit code is 0 only when no check FAILed; UNVERIFIED is allowed but visible.

Raw claude stream output for every spawned turn is kept under
/tmp/context-mode-hooks-smoke/<run-id>/ for evidence.

Usage (inside the governor container):

    python3 scripts/context_mode_hooks_smoke.py
    python3 scripts/context_mode_hooks_smoke.py --checks plugin_installed,no_duplicate_mcp
    python3 scripts/context_mode_hooks_smoke.py --timeout-sec 300 --out /tmp/smoke.jsonl

Check 7 (disable_restores_baseline) only runs for real when the operator
re-runs the script with HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED off.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Allow running as `python3 scripts/context_mode_hooks_smoke.py` from repo root
# (or anywhere) without an installed orion package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.fcc.claude_spawn import claude_permission_argv  # noqa: E402
from orion.harness.fcc_motor import (  # noqa: E402
    _build_subprocess_env,
    _preflight_fcc_server,
    expand_env_path,
    label_to_claude_model_id,
    load_fcc_env,
    resolve_auth_token,
)

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIED = "UNVERIFIED"
_STATUS_RANK = {PASS: 0, UNVERIFIED: 1, FAIL: 2}

HOOKS_FLAG_KEY = "HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED"
PLUGIN_MCP_SERVER = "mcp__plugin_context-mode_context-mode"
PLUGIN_TOOL_PREFIX = PLUGIN_MCP_SERVER + "__"
DEFAULT_CONTEXT_MODE_DIR = "/var/lib/orion/context-mode"
EVIDENCE_ROOT = Path("/tmp/context-mode-hooks-smoke")
# Smoke turns are read-only by contract: the script must never modify repo files.
SAFE_DISALLOWED_TOOLS = ["--disallowedTools", "Write", "Edit", "NotebookEdit", "Bash"]

CHECK_ORDER = [
    "plugin_installed",
    "no_duplicate_mcp",
    "plugin_tools_callable_headless",
    "precompact_fires",
    "sessionstart_restore",
    "fresh_session_isolation",
    "disable_restores_baseline",
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no subprocess, no environment access)
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check: str
    status: str
    evidence: Any

    def to_json_line(self) -> str:
        return json.dumps(
            {"check": self.check, "status": self.status, "evidence": self.evidence},
            ensure_ascii=False,
        )


def is_truthy(raw: Optional[str]) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def parse_stream_lines(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """Parse claude --output-format stream-json lines into dict events."""
    events: List[Dict[str, Any]] = []
    for line in lines:
        stripped = str(line or "").strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def extract_tool_use_names(events: Sequence[Mapping[str, Any]]) -> List[str]:
    """Tool names from assistant events' message.content[] tool_use blocks."""
    names: List[str] = []
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("name"), str)
            ):
                names.append(block["name"])
    return names


def extract_session_id(events: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """session_id from the result event (fall back to any event carrying one)."""
    fallback: Optional[str] = None
    for event in events:
        sid = event.get("session_id")
        if not sid:
            continue
        if str(event.get("type") or "") == "result":
            return str(sid)
        if fallback is None:
            fallback = str(sid)
    return fallback


def extract_final_text(events: Sequence[Mapping[str, Any]]) -> str:
    """Final reply text: prefer the result event, else last assistant text."""
    assistant_text = ""
    for event in events:
        if str(event.get("type") or "") == "result":
            result = event.get("result")
            if isinstance(result, str) and result.strip():
                return result.strip()
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if parts:
            assistant_text = "".join(parts).strip() or assistant_text
    return assistant_text


def has_compact_event(events: Sequence[Mapping[str, Any]]) -> bool:
    """Any stream event whose type/subtype mentions compaction."""
    for event in events:
        etype = str(event.get("type") or "").lower()
        subtype = str(event.get("subtype") or event.get("system_subtype") or "").lower()
        if "compact" in etype or "compact" in subtype:
            return True
    return False


def snapshot_dir(root: Path) -> Dict[str, float]:
    """relative-path -> mtime for every file under root ({} if missing)."""
    if not root.is_dir():
        return {}
    out: Dict[str, float] = {}
    for path in root.rglob("*"):
        try:
            if path.is_file():
                out[str(path.relative_to(root))] = path.stat().st_mtime
        except OSError:
            continue
    return out


def dir_diff(before: Mapping[str, float], after: Mapping[str, float]) -> Dict[str, List[str]]:
    added = sorted(name for name in after if name not in before)
    modified = sorted(
        name for name in after if name in before and after[name] > before[name]
    )
    return {"added": added, "modified": modified}


def worst_status(statuses: Iterable[str]) -> str:
    """FAIL beats UNVERIFIED beats PASS; empty input is PASS."""
    worst = PASS
    for status in statuses:
        rank = _STATUS_RANK.get(status, _STATUS_RANK[FAIL])
        if rank > _STATUS_RANK[worst]:
            worst = FAIL if rank == _STATUS_RANK[FAIL] else UNVERIFIED
    return worst


def exit_code_for(statuses: Iterable[str]) -> int:
    """0 only when no FAIL; UNVERIFIED does not fail the run."""
    return 1 if worst_status(statuses) == FAIL else 0


# ---------------------------------------------------------------------------
# Subprocess orchestration (container-only; never exercised by unit tests)
# ---------------------------------------------------------------------------


@dataclass
class SmokeConfig:
    claude_bin: str
    workspace: str
    context_mode_dir: Path
    env_path: Path
    evidence_dir: Path
    timeout_sec: float
    fcc_server_url: str = ""
    auth_token: str = ""
    model_id: str = ""
    turn_blocker: Optional[str] = None  # reason claude turns cannot be spawned


@dataclass
class TurnResult:
    events: List[Dict[str, Any]] = field(default_factory=list)
    exit_code: int = 1
    stderr_tail: str = ""
    stdout_path: Optional[Path] = None
    timed_out: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stderr_tail": self.stderr_tail[-500:],
            "raw_stream": str(self.stdout_path) if self.stdout_path else None,
        }


def build_config(args: argparse.Namespace) -> SmokeConfig:
    env_path = expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env"))
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    evidence_dir = EVIDENCE_ROOT / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cfg = SmokeConfig(
        claude_bin=os.environ.get("HARNESS_FCC_CLAUDE_BIN", "claude"),
        workspace=os.environ.get("HARNESS_FCC_WORKSPACE") or os.getcwd(),
        context_mode_dir=Path(
            os.environ.get("HARNESS_FCC_CONTEXT_MODE_DIR") or DEFAULT_CONTEXT_MODE_DIR
        ),
        env_path=env_path,
        evidence_dir=evidence_dir,
        timeout_sec=args.timeout_sec,
    )

    fcc_env = load_fcc_env(env_path)
    cfg.fcc_server_url = str(os.environ.get("HARNESS_FCC_SERVER_URL") or "").strip()
    cfg.auth_token = resolve_auth_token(fcc_env)
    problems: List[str] = []
    if not cfg.fcc_server_url:
        problems.append("HARNESS_FCC_SERVER_URL is not set")
    if not cfg.auth_token:
        problems.append(f"ANTHROPIC_AUTH_TOKEN missing from FCC env at {env_path}")
    try:
        cfg.model_id = label_to_claude_model_id(args.model_label, fcc_env)
    except ValueError as exc:
        problems.append(str(exc))
    if not problems:
        try:
            _preflight_fcc_server(cfg.fcc_server_url)
        except RuntimeError as exc:
            problems.append(str(exc))
    if problems:
        cfg.turn_blocker = "; ".join(problems)
    return cfg


def run_claude_turn(
    cfg: SmokeConfig,
    *,
    turn_name: str,
    prompt: str,
    allow_plugin_tools: bool = False,
    extra_argv: Optional[Sequence[str]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
) -> TurnResult:
    """Mirror the FCC motor spawn shape for one headless claude -p turn."""
    argv = [cfg.claude_bin, "-p", prompt, "--output-format", "stream-json", "--verbose"]
    argv += claude_permission_argv(auto_approve=True)  # root container -> dontAsk
    argv += ["--model", cfg.model_id]
    if allow_plugin_tools:
        argv += ["--allowedTools", PLUGIN_MCP_SERVER]
    argv += SAFE_DISALLOWED_TOOLS
    if extra_argv:
        argv += list(extra_argv)

    env = _build_subprocess_env(fcc_server_url=cfg.fcc_server_url, auth_token=cfg.auth_token)
    if env_overrides:
        env.update(env_overrides)

    stdout_path = cfg.evidence_dir / f"{turn_name}.stdout.jsonl"
    stderr_path = cfg.evidence_dir / f"{turn_name}.stderr.txt"
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            cwd=cfg.workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_sec,
        )
        stdout, stderr, exit_code = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\n[smoke] turn timed out after {cfg.timeout_sec}s"
        exit_code, timed_out = 124, True
    except FileNotFoundError:
        stdout, stderr, exit_code = "", f"claude binary not found: {cfg.claude_bin!r}", 127

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return TurnResult(
        events=parse_stream_lines(stdout.splitlines()),
        exit_code=exit_code,
        stderr_tail=stderr.strip()[-500:],
        stdout_path=stdout_path,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# Checks (§9.3)
# ---------------------------------------------------------------------------


def check_plugin_installed(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """1. `claude plugin list` output contains "context-mode"."""
    argv = [cfg.claude_bin, "plugin", "list"]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=min(60.0, cfg.timeout_sec)
        )
    except FileNotFoundError:
        return CheckResult(
            "plugin_installed", FAIL, f"claude binary not found: {cfg.claude_bin!r}"
        )
    except subprocess.TimeoutExpired:
        return CheckResult("plugin_installed", FAIL, "`claude plugin list` timed out")
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    matches = [line.strip() for line in output.splitlines() if "context-mode" in line.lower()]
    if proc.returncode == 0 and matches:
        return CheckResult(
            "plugin_installed", PASS, {"exit_code": proc.returncode, "matches": matches[:5]}
        )
    return CheckResult(
        "plugin_installed",
        FAIL,
        {
            "exit_code": proc.returncode,
            "detail": "no 'context-mode' entry in `claude plugin list`",
            "output_tail": output.strip()[-500:],
        },
    )


def check_no_duplicate_mcp(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """2. Hook-mode contract: rendered MCP config has no standalone context-mode."""
    from orion.fcc.mcp_config import McpPreflightError, cleanup_mcp_config, render_mcp_config

    fcc_env = load_fcc_env(cfg.env_path)
    try:
        # Render into the evidence dir but delete immediately: the rendered
        # config carries secrets (GITHUB_PAT / FIRECRAWL_API_KEY).
        path = render_mcp_config(
            correlation_id="context-mode-hooks-smoke",
            fcc_env=fcc_env,
            tmp_dir=cfg.evidence_dir / "mcp-render",
            include_context_mode=False,
        )
    except McpPreflightError as exc:
        return CheckResult(
            "no_duplicate_mcp",
            UNVERIFIED,
            f"could not render harness MCP config ({exc.error_code}): {exc}",
        )
    try:
        servers = sorted((json.loads(path.read_text(encoding="utf-8")).get("mcpServers") or {}))
    finally:
        cleanup_mcp_config(path)

    if "context-mode" in servers:
        return CheckResult(
            "no_duplicate_mcp",
            FAIL,
            {
                "detail": "standalone context-mode MCP server present despite "
                "include_context_mode=False (would double-register ctx_* tools)",
                "mcp_servers": servers,
            },
        )
    flag_raw = os.environ.get(HOOKS_FLAG_KEY)
    if is_truthy(flag_raw):
        return CheckResult(
            "no_duplicate_mcp",
            PASS,
            {"mcp_servers": servers, HOOKS_FLAG_KEY: flag_raw},
        )
    return CheckResult(
        "no_duplicate_mcp",
        UNVERIFIED,
        {
            "detail": f"rendered config is clean ({servers}) but {HOOKS_FLAG_KEY} is not "
            "truthy in this environment, so hook mode is not what live turns exercise. "
            f"Re-run with {HOOKS_FLAG_KEY}=true exported (governor compose contract).",
            HOOKS_FLAG_KEY: flag_raw or "",
        },
    )


def check_plugin_tools_callable(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """3. A headless turn can call a plugin-owned ctx_* MCP tool."""
    if cfg.turn_blocker:
        return CheckResult("plugin_tools_callable_headless", UNVERIFIED, cfg.turn_blocker)
    turn = run_claude_turn(
        cfg,
        turn_name="03_plugin_tools",
        prompt="Call the ctx_stats tool once and reply DONE.",
        allow_plugin_tools=True,
    )
    names = extract_tool_use_names(turn.events)
    plugin_calls = [n for n in names if n.startswith(PLUGIN_TOOL_PREFIX)]
    session_id = extract_session_id(turn.events)
    if session_id:
        state["session_id"] = session_id
    evidence = {
        "plugin_tool_calls": plugin_calls,
        "all_tool_calls": names,
        "session_id": session_id,
        **turn.summary(),
    }
    if plugin_calls:
        return CheckResult("plugin_tools_callable_headless", PASS, evidence)
    return CheckResult("plugin_tools_callable_headless", FAIL, evidence)


def check_precompact_fires(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """4. Forced-low context turn triggers PreCompact snapshots (or compact event)."""
    if cfg.turn_blocker:
        return CheckResult("precompact_fires", UNVERIFIED, cfg.turn_blocker)
    sessions_dir = cfg.context_mode_dir / "sessions"
    before = snapshot_dir(sessions_dir)
    turn = run_claude_turn(
        cfg,
        turn_name="04_precompact",
        prompt=(
            "Read README.md in full, then read CLAUDE.md in full, then reply with a "
            "one-line summary of each."
        ),
        allow_plugin_tools=True,
        env_overrides={
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "16384",
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "16384",
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "10",
        },
    )
    after = snapshot_dir(sessions_dir)
    diff = dir_diff(before, after)
    compacted = has_compact_event(turn.events)
    evidence = {
        "sessions_dir": str(sessions_dir),
        "snapshot_diff": diff,
        "compact_event_in_stream": compacted,
        **turn.summary(),
    }
    if diff["added"] or diff["modified"] or compacted:
        return CheckResult("precompact_fires", PASS, evidence)
    evidence["detail"] = (
        "no new/updated snapshot under sessions/ and no compact event in the stream; "
        "compaction may not have triggered (context did not fill past the override, "
        "or the PreCompact hook did not run). Inspect the raw stream for tool_use "
        "volume and re-run with a heavier prompt if needed."
    )
    return CheckResult("precompact_fires", UNVERIFIED, evidence)


def check_sessionstart_restore(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """5. --resume of check 3's session recalls the ctx_stats call."""
    if cfg.turn_blocker:
        return CheckResult("sessionstart_restore", UNVERIFIED, cfg.turn_blocker)
    session_id = state.get("session_id")
    if not session_id:
        return CheckResult(
            "sessionstart_restore",
            UNVERIFIED,
            "no session_id captured — run the plugin_tools_callable_headless check "
            "first (same invocation) so there is a session to resume",
        )
    turn = run_claude_turn(
        cfg,
        turn_name="05_resume",
        prompt="In one line, what tool did you call earlier in this session?",
        allow_plugin_tools=True,
        extra_argv=["--resume", str(session_id)],
    )
    final = extract_final_text(turn.events)
    evidence = {"resumed_session_id": session_id, "final_text": final[:500], **turn.summary()}
    if turn.exit_code != 0:
        evidence["detail"] = (
            "`claude -p --resume <session_id>` exited nonzero; the flag may be "
            "rejected in this CLI version — see stderr_tail"
        )
        return CheckResult("sessionstart_restore", UNVERIFIED, evidence)
    if "ctx_stats" in final.lower():
        return CheckResult("sessionstart_restore", PASS, evidence)
    evidence["detail"] = "resumed reply does not reference ctx_stats"
    return CheckResult("sessionstart_restore", FAIL, evidence)


def check_fresh_session_isolation(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """6. A brand-new session must not leak check 3's markers."""
    if cfg.turn_blocker:
        return CheckResult("fresh_session_isolation", UNVERIFIED, cfg.turn_blocker)
    turn = run_claude_turn(
        cfg,
        turn_name="06_fresh_session",
        prompt=(
            "Without calling any tool, state in one line what task you were working "
            "on before this message."
        ),
    )
    final = extract_final_text(turn.events)
    markers = ["ctx_stats", "ctx stats"]
    leaked = [m for m in markers if m in final.lower()]
    evidence = {"final_text": final[:500], "leaked_markers": leaked, **turn.summary()}
    if turn.exit_code != 0 and not final:
        evidence["detail"] = "fresh turn exited nonzero with no reply — see stderr_tail"
        return CheckResult("fresh_session_isolation", UNVERIFIED, evidence)
    if leaked:
        evidence["detail"] = "fresh session reply mentions markers from the earlier smoke turn"
        return CheckResult("fresh_session_isolation", FAIL, evidence)
    return CheckResult("fresh_session_isolation", PASS, evidence)


def check_disable_restores_baseline(cfg: SmokeConfig, state: Dict[str, Any]) -> CheckResult:
    """7. With the hooks flag off, plain turns show no plugin tool_use."""
    flag_raw = os.environ.get(HOOKS_FLAG_KEY)
    if is_truthy(flag_raw):
        return CheckResult(
            "disable_restores_baseline",
            UNVERIFIED,
            f"{HOOKS_FLAG_KEY} is truthy ({flag_raw!r}) in this run; re-run this "
            f"script with the flag off (e.g. {HOOKS_FLAG_KEY}=false) to verify the "
            "baseline turn shape is restored",
        )
    if cfg.turn_blocker:
        return CheckResult("disable_restores_baseline", UNVERIFIED, cfg.turn_blocker)
    turn = run_claude_turn(
        cfg,
        turn_name="07_baseline",
        prompt="Reply with the single word OK.",
    )
    names = extract_tool_use_names(turn.events)
    plugin_calls = [n for n in names if n.startswith("mcp__plugin_context-mode")]
    evidence = {"plugin_tool_calls": plugin_calls, "all_tool_calls": names, **turn.summary()}
    if turn.exit_code != 0 and not turn.events:
        evidence["detail"] = "baseline turn produced no stream — see stderr_tail"
        return CheckResult("disable_restores_baseline", UNVERIFIED, evidence)
    if plugin_calls:
        evidence["detail"] = "plugin-owned context-mode tools still fired with the flag off"
        return CheckResult("disable_restores_baseline", FAIL, evidence)
    return CheckResult("disable_restores_baseline", PASS, evidence)


CHECKS = {
    "plugin_installed": check_plugin_installed,
    "no_duplicate_mcp": check_no_duplicate_mcp,
    "plugin_tools_callable_headless": check_plugin_tools_callable,
    "precompact_fires": check_precompact_fires,
    "sessionstart_restore": check_sessionstart_restore,
    "fresh_session_isolation": check_fresh_session_isolation,
    "disable_restores_baseline": check_disable_restores_baseline,
}
assert list(CHECKS) == CHECK_ORDER


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Headless Context Mode hook-validation smoke (semantic self-indexing "
            "§9.3 Stage B). Emits one JSON line per check; exits 0 only if no FAIL."
        )
    )
    parser.add_argument(
        "--checks",
        default=",".join(CHECK_ORDER),
        help=f"comma-separated subset of checks to run (default: all). Known: {', '.join(CHECK_ORDER)}",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=240.0,
        help="per-claude-turn timeout in seconds (default: 240)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="optional path: also append each check's JSON line to this JSONL file",
    )
    parser.add_argument(
        "--model-label",
        default="MODEL_SONNET",
        help="FCC env model label used for spawned turns (default: MODEL_SONNET)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected = [name.strip() for name in str(args.checks).split(",") if name.strip()]
    unknown = [name for name in selected if name not in CHECKS]
    if unknown:
        print(
            f"unknown checks: {', '.join(unknown)} (known: {', '.join(CHECK_ORDER)})",
            file=sys.stderr,
        )
        return 2
    # Preserve canonical order regardless of how --checks was written.
    selected = [name for name in CHECK_ORDER if name in selected]

    cfg = build_config(args)
    print(f"[smoke] evidence dir: {cfg.evidence_dir}", file=sys.stderr)
    if cfg.turn_blocker:
        print(
            f"[smoke] claude turns blocked (turn checks will be UNVERIFIED): {cfg.turn_blocker}",
            file=sys.stderr,
        )

    out_path = Path(args.out) if args.out else None
    state: Dict[str, Any] = {}
    results: List[CheckResult] = []
    for name in selected:
        try:
            result = CHECKS[name](cfg, state)
        except Exception as exc:  # honest failure beats a crashed smoke
            result = CheckResult(name, FAIL, f"check crashed: {type(exc).__name__}: {exc}")
        results.append(result)
        line = result.to_json_line()
        print(line, flush=True)
        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    statuses = [r.status for r in results]
    print(
        f"[smoke] overall: {worst_status(statuses)} "
        f"({statuses.count(PASS)} PASS / {statuses.count(UNVERIFIED)} UNVERIFIED / "
        f"{statuses.count(FAIL)} FAIL); evidence dir: {cfg.evidence_dir}",
        file=sys.stderr,
    )
    return exit_code_for(statuses)


if __name__ == "__main__":
    sys.exit(main())

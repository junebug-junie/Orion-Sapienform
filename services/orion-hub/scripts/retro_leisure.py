"""Solo leisure: Orion plays alone when idle and the cabinet is cool (spec section E).

Spec: docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md.

There is no scheduler here. The allocator admits a ``play`` act (Lane C), the
``retro_leisure_session`` dispatch consumer calls ``should_start(config)`` and, when that
returns None, ``run_leisure_session(config, act_id=...)``. The runner re-checks the gate
itself (time passes between the two calls).

Doer-side gates (``leisure_block_reason``), checked at start and every RECHECK_SEC:
  disabled / retro_disabled / game_not_allowed     RETRO_LEISURE_ENABLED, HUB_RETRO_ENABLED, allowlist
  arousal_unknown / cabinet_hot / cabinet_unknown / arousal_strained
                                                   orion:regulation:latest (fail closed)
  presence_unknown / browser_connected / human_recent   (start) no relay socket open and no
                                                   human Retro action for HUMAN_QUIET_SEC
  human_present                                    (running) a human acted after we started
  retro_session_active / retro_unavailable         (start) never replaces anyone's game
  ledger_unknown / cooldown / daily_cap / session_cap   one start per MIN_GAP_SEC, DAILY_MAX_SEC
                                                   per rolling 24 h, SESSION_MAX_SEC per session
A browser that only WATCHES (media socket) does not end a running session: opening the page
shows it live. A human acquire or takeover is preempted by Retro's arbiter (Lane A); the
runner sees its policy gone and ends ``human_preempted`` without stopping the human's game.

The runner is a server-side sponsor: it holds a ``leisure_id`` grant instead of a browser
connection. Retro contract used (Lane A; every call goes over Retro's UDS with hub.token):
  GET  /health/ready, /games/{game}/saves, /sessions/{sid}, /sessions/{sid}/atlas
  POST /sessions {game_id, idempotency_key, save_id?}
  POST /sessions/{sid}/scripted {leisure_id, generation, port, action, duration_seconds, cognition}
  POST /sessions/{sid}/goal  GoalV2 {leisure_id, generation, port, expected_epoch, decision_id,
                                     goal, prediction, rationale, planner_id, budget_seconds}
  POST /sessions/{sid}/save, /sessions/{sid}/stop
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from scripts import retro_planner
from scripts.retro_planner import PlannerBusy, PlannerRejected, PlannerShed
from scripts.retro_session_end import RetroSessionEnded

logger = logging.getLogger('orion-hub.retro.leisure')

RECHECK_SEC = 30.0
POLL_SEC = 1.0
HUMAN_QUIET_SEC = 600.0
MAX_CONSECUTIVE_SHEDS = 3
MAX_CONSECUTIVE_FAILURES = 3
FAILURE_RETRY_SEC = 5.0
DAY_SEC = 86400.0
START_TIMEOUT_SEC = 10.0
ACTIVE_STATES = ('starting', 'running', 'paused')


# --- the gate (pure, deterministic) -----------------------------------------------------------

@dataclass(frozen=True)
class LeisureGateInputs:
    now: float
    enabled: bool
    retro_enabled: bool
    game_id: str
    allowed_games: tuple
    session_max_sec: float
    daily_max_sec: float
    min_gap_sec: float
    arousal_level: str = 'unknown'            # engaged | idle | strained | unknown
    arousal_reason: str = 'absent'
    cabinet_reflex: Optional[str] = None      # cabinet_hot | cabinet_unknown | None (clear)
    presence_read_ok: bool = False
    seconds_since_human: Optional[float] = None   # None with read_ok: no human action in a day
    browsers_connected: Optional[int] = None
    retro_session_state: Optional[str] = 'unreadable'   # None: no session; 'unreadable': failed
    ledger: Optional[list] = None
    human_quiet_sec: float = HUMAN_QUIET_SEC
    # Set while a session runs:
    own_leisure_id: Optional[str] = None
    elapsed_sec: float = 0.0


def _used_sec(inputs: LeisureGateInputs) -> float:
    used = 0.0
    for row in inputs.ledger or []:
        if not isinstance(row, dict) or row.get('leisure_id') == inputs.own_leisure_id:
            continue
        started = float(row.get('started_at') or 0)
        if inputs.now - started > DAY_SEC:
            continue
        if row.get('ended_at') is None:   # a crashed runner: charge up to the session cap
            used += min(max(0.0, inputs.now - started), inputs.session_max_sec)
        else:
            used += max(0.0, float(row.get('seconds') or 0))
    return used


def leisure_block_reason(inputs: LeisureGateInputs) -> Optional[str]:
    """The named reason leisure may not start (or must end), or None. No I/O."""
    running = inputs.own_leisure_id is not None
    if not inputs.enabled:
        return 'disabled'
    if not inputs.retro_enabled:
        return 'retro_disabled'
    if inputs.game_id not in inputs.allowed_games:
        return 'game_not_allowed'
    if inputs.arousal_level == 'unknown':
        return 'arousal_unknown'
    if inputs.cabinet_reflex == 'cabinet_hot':
        return 'cabinet_hot'
    if inputs.cabinet_reflex:
        return 'cabinet_unknown' if inputs.cabinet_reflex == 'cabinet_unknown' else 'cabinet_' + str(inputs.cabinet_reflex)
    if inputs.arousal_level == 'strained':
        return 'arousal_strained'
    if not inputs.presence_read_ok:
        return 'presence_unknown'
    if running:
        if inputs.seconds_since_human is not None and inputs.seconds_since_human < inputs.elapsed_sec:
            return 'human_present'
    else:
        if inputs.browsers_connected is None:
            return 'presence_unknown'
        if inputs.browsers_connected > 0:
            return 'browser_connected'
        if inputs.seconds_since_human is not None and inputs.seconds_since_human < inputs.human_quiet_sec:
            return 'human_recent'
        if inputs.retro_session_state == 'unreadable':
            return 'retro_unavailable'
        if inputs.retro_session_state in ACTIVE_STATES:
            return 'retro_session_active'
    if inputs.ledger is None:
        return 'ledger_unknown'
    used = _used_sec(inputs)
    if running:
        if inputs.elapsed_sec >= inputs.session_max_sec:
            return 'session_cap'
        if used + inputs.elapsed_sec >= inputs.daily_max_sec:
            return 'daily_cap'
        return None
    starts = [float(r.get('started_at') or 0) for r in inputs.ledger if isinstance(r, dict)]
    if starts and inputs.now - max(starts) < inputs.min_gap_sec:
        return 'cooldown'
    if used >= inputs.daily_max_sec:
        return 'daily_cap'
    return None


# --- I/O ----------------------------------------------------------------------------------------

class RetroClient:
    """Retro over its UDS, with Hub's credential. Each call returns (status, json)."""

    def __init__(self, config):
        self.config = config

    def _headers(self):
        token = Path(self.config.HUB_RETRO_AUTH_DIR, 'hub.token').read_text().strip()
        if len(token) < 32:
            raise RuntimeError('retro_credential_missing')
        return {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

    async def call(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, Any]:
        import aiohttp
        async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=self.config.HUB_RETRO_SOCKET),
                                         timeout=aiohttp.ClientTimeout(total=5)) as c:
            async with c.request(method, 'http://retro/' + path, headers=self._headers(),
                                 data=None if body is None else json.dumps(body)) as r:
                try:
                    data = await r.json(content_type=None)
                except Exception:  # noqa: BLE001
                    data = None
                return r.status, data


async def read_regulation(redis: Any, *, max_age_sec: float, now=None) -> tuple[str, str, Optional[str]]:
    """(arousal level, reason, cabinet_reflex) from one GET of orion:regulation:latest.
    Anything doubtful is ``unknown`` (and the gate refuses on unknown)."""
    from datetime import datetime, timezone
    from orion.regulation.arousal_view import arousal_view
    from orion.schemas.regulation import REGULATION_STATE_REDIS_KEY
    try:
        raw = await asyncio.wait_for(redis.get(REGULATION_STATE_REDIS_KEY), 2.0)
    except Exception:  # noqa: BLE001
        return 'unknown', 'read_failed', None
    view = arousal_view(raw, now=now or datetime.now(timezone.utc), max_age_sec=max_age_sec)
    reflex = None
    if view.known:
        try:
            reflex = (json.loads(raw).get('arousal') or {}).get('cabinet_reflex')
        except Exception:  # noqa: BLE001
            return 'unknown', 'unparseable', None
    return view.level, view.reason, reflex


@dataclass
class LeisureDeps:
    store: Any                                    # scripts.retro_presence.PresenceStore
    retro: Any                                    # RetroClient-like: async call(method, path, body)
    regulation: Callable[[], Awaitable[tuple]]    # -> (level, reason, cabinet_reflex)
    plan: Callable[..., Awaitable[dict]]          # scripts.retro_planner.plan_zelda signature
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def default_deps(config) -> LeisureDeps:
    from scripts.retro_presence import store_for
    store = store_for(config)
    if store is None:
        raise RuntimeError('leisure_requires_bus_url')
    max_age = float(getattr(config, 'ORION_REGULATION_READER_MAX_AGE_SEC', 360.0))
    return LeisureDeps(store=store, retro=RetroClient(config),
                       regulation=lambda: read_regulation(store.redis, max_age_sec=max_age),
                       plan=lambda *a, **k: retro_planner.plan_zelda(config, *a, **k))


def _settings(config) -> dict:
    games = getattr(config, 'RETRO_LEISURE_GAME_IDS', '') or ''
    allowed = tuple(g.strip() for g in (games.split(',') if isinstance(games, str) else games) if g.strip())
    return dict(enabled=bool(getattr(config, 'RETRO_LEISURE_ENABLED', False)),
                retro_enabled=bool(getattr(config, 'HUB_RETRO_ENABLED', False)),
                allowed_games=allowed, game_id=allowed[0] if allowed else '',
                session_max_sec=float(getattr(config, 'RETRO_LEISURE_SESSION_MAX_SEC', 1800)),
                daily_max_sec=float(getattr(config, 'RETRO_LEISURE_DAILY_MAX_SEC', 5400)),
                min_gap_sec=float(getattr(config, 'RETRO_LEISURE_MIN_GAP_SEC', 10800)),
                port=int(getattr(config, 'RETRO_LEISURE_PORT', 0)))


async def gather_gate_inputs(config, deps: LeisureDeps, *, own_leisure_id=None, elapsed_sec=0.0) -> LeisureGateInputs:
    s = _settings(config)
    base = LeisureGateInputs(now=deps.clock(), enabled=s['enabled'], retro_enabled=s['retro_enabled'],
                             game_id=s['game_id'], allowed_games=s['allowed_games'],
                             session_max_sec=s['session_max_sec'], daily_max_sec=s['daily_max_sec'],
                             min_gap_sec=s['min_gap_sec'], own_leisure_id=own_leisure_id,
                             elapsed_sec=elapsed_sec)
    if not (s['enabled'] and s['retro_enabled']):
        return base   # switched off: refuse without touching Redis or Retro
    level, reason, reflex = await deps.regulation()
    read_ok, since_human = await deps.store.seconds_since_human()
    browsers = await deps.store.browsers_connected()
    ledger = await deps.store.ledger()
    state: Optional[str] = 'unreadable'
    if own_leisure_id is None:
        try:
            status, ready = await deps.retro.call('GET', 'health/ready')
            if status == 200 and isinstance(ready, dict):
                session = ready.get('session')
                state = session.get('state') if isinstance(session, dict) else None
        except Exception:  # noqa: BLE001
            state = 'unreadable'
    return replace(base, arousal_level=level, arousal_reason=reason, cabinet_reflex=reflex,
                   presence_read_ok=read_ok, seconds_since_human=since_human,
                   browsers_connected=browsers, retro_session_state=state, ledger=ledger)


async def should_start(config, deps: Optional[LeisureDeps] = None) -> Optional[str]:
    """Lane C's dispatch consumer calls this before ``run_leisure_session``. None = may start."""
    s = _settings(config)
    if not s['enabled']:
        return 'disabled'
    if not s['retro_enabled']:
        return 'retro_disabled'
    try:
        deps = deps or default_deps(config)
    except Exception as exc:  # noqa: BLE001
        return 'presence_unknown:' + str(exc)[:80]
    return leisure_block_reason(await gather_gate_inputs(config, deps))


# --- the runner -----------------------------------------------------------------------------------

@dataclass
class LeisureOutcome:
    started: bool
    refusal: Optional[str] = None
    ended: Optional[RetroSessionEnded] = None


@dataclass
class _Run:
    leisure_id: str
    act_id: Optional[str]
    game_id: str
    port: int
    started_at: float
    session_max_sec: float = 1800.0
    session_id: Optional[str] = None
    generation: Optional[str] = None
    epoch: Optional[int] = None
    lineage_id: Optional[str] = None
    loaded_save_id: Optional[str] = None
    planned: int = 0
    rejections: int = 0
    sheds: int = 0
    consecutive_sheds: int = 0
    consecutive_failures: int = 0
    next_plan_at: float = 0.0
    planned_decision: Optional[str] = None
    waiting: Optional[dict] = None
    last_goal: Optional[dict] = None
    verdicts: dict = field(default_factory=lambda: {'met': 0, 'refuted': 0, 'unknown': 0})
    history: list = field(default_factory=list)


_running_here = False


async def _journal(deps: LeisureDeps, event: str, **data) -> None:
    row = dict(event=event, at=round(deps.clock(), 3), **data)
    logger.info('retro_leisure %s %s', event, json.dumps(data, separators=(',', ':'), default=str))
    try:
        await deps.store.append_journal(row)
    except Exception:  # noqa: BLE001 -- the log line above is the fallback trace, not a decision
        logger.warning('retro_leisure_journal_write_failed event=%s', event, exc_info=True)


async def _status(deps: LeisureDeps, run: Optional[_Run], state: str, **extra) -> None:
    status = dict(state=state, updated_at=round(deps.clock(), 3), **extra)
    if run:
        status |= dict(leisure_id=run.leisure_id, act_id=run.act_id, session_id=run.session_id,
                       game_id=run.game_id, port=run.port, lineage_id=run.lineage_id,
                       started_at=run.started_at, decisions=run.planned, sheds=run.sheds,
                       verdicts=dict(run.verdicts), waiting=run.waiting, last_goal=run.last_goal,
                       history=run.history[-5:])
    try:
        await deps.store.write_status(status)
    except Exception:  # noqa: BLE001
        logger.warning('retro_leisure_status_write_failed state=%s', state, exc_info=True)


def _policy(session: dict, run: _Run) -> Optional[dict]:
    for p in session.get('scripted') or []:
        if not isinstance(p, dict) or p.get('port') != run.port:
            continue
        if p.get('leisure_id') not in (None, run.leisure_id):
            return None
        if run.epoch is None or p.get('epoch') == run.epoch:
            return p
    return None


def _count_verdicts(history: Any) -> dict:
    out = {'met': 0, 'refuted': 0, 'unknown': 0}
    for h in history or []:
        if isinstance(h, dict):
            v = h.get('verdict')
            out[v if v in out else 'unknown'] += 1
    return out


async def run_leisure_session(config, *, act_id: Optional[str] = None,
                              deps: Optional[LeisureDeps] = None) -> LeisureOutcome:
    """Run one leisure session to its end. Returns a refusal, or the ended-session data for
    the memory export (``outcome.ended``, see scripts/retro_session_end.py)."""
    global _running_here
    s = _settings(config)
    if not s['enabled']:
        return LeisureOutcome(started=False, refusal='disabled')
    deps = deps or default_deps(config)
    if _running_here:
        await _journal(deps, 'refused', reason='leisure_already_running', act_id=act_id)
        return LeisureOutcome(started=False, refusal='leisure_already_running')
    inputs = await gather_gate_inputs(config, deps)
    reason = leisure_block_reason(inputs)
    if reason:
        await _journal(deps, 'refused', reason=reason, act_id=act_id, arousal=inputs.arousal_reason)
        await _status(deps, None, 'refused', reason=reason, act_id=act_id)
        return LeisureOutcome(started=False, refusal=reason)
    run = _Run(leisure_id=str(uuid.uuid4()), act_id=act_id, game_id=s['game_id'], port=s['port'],
               started_at=deps.clock(), session_max_sec=s['session_max_sec'])
    lock_ttl = int(s['session_max_sec'] + 300)
    try:
        locked = await deps.store.acquire_lock(run.leisure_id, lock_ttl)
    except Exception:  # noqa: BLE001
        locked = False
    if not locked:
        await _journal(deps, 'refused', reason='leisure_already_running', act_id=act_id)
        return LeisureOutcome(started=False, refusal='leisure_already_running')
    _running_here = True
    try:
        return await _run(config, deps, s, run, inputs.ledger or [], lock_ttl)
    finally:
        _running_here = False
        try:
            await deps.store.release_lock(run.leisure_id)
        except Exception:  # noqa: BLE001
            logger.warning('retro_leisure_lock_release_failed', exc_info=True)


async def _start_game(deps: LeisureDeps, run: _Run) -> Optional[str]:
    """Load the newest compatible save (the lineage carries over) or start fresh. Returns a
    refusal reason or None."""
    status, saves = await deps.retro.call('GET', f'games/{run.game_id}/saves')
    if status == 200 and isinstance(saves, list):
        compatible = [x for x in saves if isinstance(x, dict) and x.get('compatible')]
        run.loaded_save_id = compatible[0]['save_id'] if compatible else None
    elif status not in (200, 409):   # 409 saves_unavailable: play fresh, no lineage carry-over
        return f'retro_saves_unreadable:{status}'
    body = {'game_id': run.game_id, 'idempotency_key': run.leisure_id.replace('-', '')}
    if run.loaded_save_id:
        body['save_id'] = run.loaded_save_id
    status, started = await deps.retro.call('POST', 'sessions', body)
    if status != 200 or not isinstance(started, dict) or not started.get('session_id'):
        detail = started.get('detail') if isinstance(started, dict) else None
        return 'retro_session_active' if detail == 'session_active' else f'retro_start_failed:{detail or status}'
    run.session_id, run.generation = started['session_id'], started.get('generation')
    deadline = deps.clock() + START_TIMEOUT_SEC
    while True:
        status, session = await deps.retro.call('GET', f'sessions/{run.session_id}')
        if status == 200 and isinstance(session, dict) and session.get('state') == 'running':
            run.generation = session.get('generation', run.generation)
            break
        if deps.clock() >= deadline or (status == 200 and session.get('state') in ('failed', 'stopped')):
            return 'retro_start_timeout'
        await deps.sleep(0.25)
    # A session paused for no_controller must be resumed before a leisure acquire (Retro README).
    await deps.retro.call('POST', f'sessions/{run.session_id}/resume', {})
    # The leisure grant: a preemptible controller lease (2 s TTL), handed straight to the script.
    status, lease = await deps.retro.call('POST', f'sessions/{run.session_id}/control', {
        'leisure_id': run.leisure_id, 'port': run.port, 'action': 'acquire'})
    if status != 200 or not isinstance(lease, dict) or not isinstance(lease.get('epoch'), int):
        detail = lease.get('detail') if isinstance(lease, dict) else status
        return f'leisure_grant_refused:{detail}'
    status, grant = await deps.retro.call('POST', f'sessions/{run.session_id}/scripted', {
        'leisure_id': run.leisure_id, 'generation': run.generation, 'port': run.port,
        'action': 'start', 'duration_seconds': int(min(1800, run.session_max_sec)), 'cognition': True,
        'expected_epoch': lease['epoch']})
    if status != 200:
        detail = grant.get('detail') if isinstance(grant, dict) else status
        return f'leisure_grant_refused:{detail}'
    status, session = await deps.retro.call('GET', f'sessions/{run.session_id}')
    policy = _policy(session, run) if status == 200 and isinstance(session, dict) else None
    if not policy or not isinstance(policy.get('epoch'), int):
        return 'leisure_grant_not_visible'
    run.epoch = policy['epoch']
    status, atlas = await deps.retro.call('GET', f'sessions/{run.session_id}/atlas')
    if status == 200 and isinstance(atlas, dict):
        run.lineage_id = atlas.get('lineage_id')
    return None


# How a session ends, from Retro's side:
#   NORMAL       Orion ended it (caps, heat, sheds, ...): release the policy, save, close the game.
#   RELEASE      a human acted but the policy still holds the port (e.g. they pressed Pause):
#                release the policy, leave their game running.
#   POLICY_GONE  Retro already dropped the policy (a human acquire/takeover, or Retro's own stop):
#                never stop the game; the human may be playing it.
# In every mode, a human action since start -- or an UNREADABLE presence -- means "a human may be
# in control": no end-save on their game and no session stop. Fail safe, never fail open.
NORMAL, RELEASE, POLICY_GONE = 'normal', 'release', 'policy_gone'
MAX_TRANSPORT_FAILURES = 5
# Retro refusing a goal because the decision moved on (room change during a long call) is play,
# not a planner failure (services/orion-retro/app/zelda/navigate.py accept()).
STALE_GOAL_REFUSALS = ('stale_decision', 'expired_decision', 'goal_active', 'stale_script_epoch',
                       'session_generation_mismatch')


async def _run(config, deps: LeisureDeps, s: dict, run: _Run, ledger: list, lock_ttl: int) -> LeisureOutcome:
    await _journal(deps, 'starting', leisure_id=run.leisure_id, act_id=run.act_id, game_id=run.game_id)
    await _status(deps, run, 'starting')
    try:
        refusal = await _start_game(deps, run)
    except Exception as exc:  # noqa: BLE001
        refusal = f'retro_unavailable:{type(exc).__name__}'
    if refusal:
        await asyncio.shield(_finish(deps, run, refusal, mode=NORMAL, ledger=None, started=False))
        return LeisureOutcome(started=False, refusal=refusal)
    ledger = [r for r in ledger if isinstance(r, dict)] + [dict(
        leisure_id=run.leisure_id, session_id=run.session_id, started_at=run.started_at,
        ended_at=None, seconds=0, act_id=run.act_id)]
    try:
        await deps.store.write_ledger(ledger)
    except Exception:  # noqa: BLE001 -- an unrecorded session must not run: the caps depend on it
        await asyncio.shield(_finish(deps, run, 'ledger_unwritable', mode=NORMAL, ledger=None))
        return LeisureOutcome(started=False, refusal='ledger_unwritable')
    await _journal(deps, 'started', leisure_id=run.leisure_id, session_id=run.session_id,
                   lineage_id=run.lineage_id, loaded_save_id=run.loaded_save_id)
    used_before = _used_sec(LeisureGateInputs(
        now=run.started_at, enabled=True, retro_enabled=True, game_id=run.game_id, allowed_games=(),
        session_max_sec=s['session_max_sec'], daily_max_sec=s['daily_max_sec'], min_gap_sec=0,
        ledger=ledger, own_leisure_id=run.leisure_id))
    try:
        end_reason, mode = await _loop(config, deps, s, run, lock_ttl, used_before)
    except asyncio.CancelledError:
        # The caller (dispatch consumer) was cancelled: still release the port and close out.
        await asyncio.shield(_finish(deps, run, 'cancelled', mode=NORMAL, ledger=ledger))
        raise
    except Exception as exc:  # noqa: BLE001 -- never leave a policy driving with nobody watching
        logger.exception('retro_leisure_loop_failed')
        end_reason, mode = f'runner_error:{type(exc).__name__}', NORMAL
    ended = await asyncio.shield(_finish(deps, run, end_reason, mode=mode, ledger=ledger))
    return LeisureOutcome(started=True, ended=ended)


async def _loop(config, deps: LeisureDeps, s: dict, run: _Run, lock_ttl: int,
                used_before: float) -> tuple[str, str]:
    plan_task: Optional[asyncio.Task] = None
    plan_decision: Optional[str] = None
    next_check = run.started_at + RECHECK_SEC
    transport_failures = 0
    try:
        while True:
            now = deps.clock()
            elapsed = now - run.started_at
            if elapsed >= s['session_max_sec']:
                return 'session_cap', NORMAL
            if used_before + elapsed >= s['daily_max_sec']:   # pure arithmetic: every poll
                return 'daily_cap', NORMAL
            try:
                status, session = await deps.retro.call('GET', f'sessions/{run.session_id}')
            except Exception:  # noqa: BLE001 -- transport error / timeout: retry a few polls
                transport_failures += 1
                if transport_failures >= MAX_TRANSPORT_FAILURES:
                    return 'retro_unreachable', NORMAL
                await deps.sleep(POLL_SEC)
                continue
            transport_failures = 0
            if (status != 200 or not isinstance(session, dict) or session.get('session_id') != run.session_id
                    or session.get('state') not in ('running', 'paused')):
                return 'retro_session_ended', POLICY_GONE
            policy = _policy(session, run)
            if policy is None:
                ok, since = await deps.store.seconds_since_human()
                if not ok:
                    return 'policy_ended_presence_unknown', POLICY_GONE
                human = since is not None and since <= deps.clock() - run.started_at
                return ('human_preempted' if human else 'policy_ended_by_retro'), POLICY_GONE
            cognition = policy.get('cognition') or {}
            if not retro_planner.is_zelda_cognition(cognition):
                return 'not_a_zelda_session', NORMAL
            retro_planner.remember_finished(run.history, cognition)
            run.verdicts = _count_verdicts(run.history)
            if now >= next_check:
                next_check = now + RECHECK_SEC
                inputs = await gather_gate_inputs(config, deps, own_leisure_id=run.leisure_id, elapsed_sec=elapsed)
                reason = leisure_block_reason(inputs)
                if reason:
                    return reason, (RELEASE if reason in ('human_present', 'presence_unknown') else NORMAL)
                try:
                    if not await deps.store.refresh_lock(run.leisure_id, lock_ttl):
                        return 'leisure_lock_lost', NORMAL
                except Exception:  # noqa: BLE001
                    return 'leisure_lock_lost', NORMAL
            if plan_task is not None and plan_task.done():
                task, plan_task = plan_task, None
                ended = await _settle_plan(deps, run, task, plan_decision)
                if ended:
                    return ended, NORMAL
            decision = cognition.get('attempt')
            decision_id = decision.get('decision_id') if isinstance(decision, dict) else None
            if (plan_task is None and decision_id and decision_id != run.planned_decision
                    and deps.clock() >= run.next_plan_at):
                plan_task = asyncio.ensure_future(deps.plan(decision, await _atlas(deps, run),
                                                            run.history[-retro_planner.HISTORY_LIMIT:]))
                plan_decision = decision_id
            await _status(deps, run, 'waiting' if run.waiting else 'playing',
                          ends_by=run.started_at + s['session_max_sec'])
            await deps.sleep(POLL_SEC)
    finally:
        if plan_task is not None and not plan_task.done():
            plan_task.cancel()
            await asyncio.gather(plan_task, return_exceptions=True)


async def _settle_plan(deps: LeisureDeps, run: _Run, task: asyncio.Task, decision_id: Optional[str]) -> Optional[str]:
    """Act on one finished planner call. Returns an end reason, or None to keep playing."""
    try:
        goal = task.result()
    except PlannerShed as exc:
        run.sheds += 1
        run.consecutive_sheds += 1
        retry = float(exc.details.get('retry_after_sec') or 30.0)
        run.waiting = dict(reason=exc.details.get('reason'), retry_after_sec=retry)
        run.next_plan_at = deps.clock() + retry
        await _journal(deps, 'planner_shed', session_id=run.session_id, decision_id=decision_id,
                       reason=exc.details.get('reason'), retry_after_sec=retry, consecutive=run.consecutive_sheds)
        return 'shed_x3' if run.consecutive_sheds >= MAX_CONSECUTIVE_SHEDS else None
    except PlannerBusy:
        run.next_plan_at = deps.clock() + POLL_SEC
        return None
    except PlannerRejected as exc:
        if exc.reason in ('observation_unknown', 'decision_pending'):
            run.next_plan_at = deps.clock() + POLL_SEC   # nothing to answer yet; no model call
            return None
        run.rejections += 1
        run.consecutive_failures += 1
        run.next_plan_at = deps.clock() + FAILURE_RETRY_SEC
        await _journal(deps, 'planner_rejected', session_id=run.session_id,
                       decision_id=decision_id, reason=exc.reason, **exc.details)
    except Exception as exc:  # noqa: BLE001 -- PlannerUnavailable and transport errors
        run.rejections += 1
        run.consecutive_failures += 1
        run.next_plan_at = deps.clock() + FAILURE_RETRY_SEC
        await _journal(deps, 'planner_failed', session_id=run.session_id, decision_id=decision_id,
                       reason=getattr(exc, 'reason', type(exc).__name__))
    else:
        run.consecutive_sheds = 0
        run.waiting = None
        body = dict(goal, leisure_id=run.leisure_id, generation=run.generation,
                    port=run.port, expected_epoch=run.epoch)
        try:
            status, accepted = await deps.retro.call('POST', f'sessions/{run.session_id}/goal', body)
        except Exception as exc:  # noqa: BLE001
            status, accepted = 0, {'detail': type(exc).__name__}
        detail = accepted.get('detail') if isinstance(accepted, dict) else None
        if status == 200:
            run.planned += 1
            run.consecutive_failures = 0
            run.planned_decision = decision_id
            run.last_goal = {k: goal.get(k) for k in ('decision_id', 'goal', 'prediction', 'rationale',
                                                      'budget_seconds', 'planner_id')}
            await _journal(deps, 'goal_accepted', session_id=run.session_id, **run.last_goal)
        elif detail in STALE_GOAL_REFUSALS:
            run.planned_decision = decision_id   # that slot is gone; wait for Retro's next offer
            await _journal(deps, 'goal_stale', session_id=run.session_id, decision_id=decision_id, detail=detail)
        else:
            run.rejections += 1
            run.consecutive_failures += 1
            run.next_plan_at = deps.clock() + FAILURE_RETRY_SEC
            await _journal(deps, 'goal_refused_by_retro', session_id=run.session_id,
                           decision_id=decision_id, status=status, detail=detail)
    return 'planner_failed_x3' if run.consecutive_failures >= MAX_CONSECUTIVE_FAILURES else None


async def _atlas(deps: LeisureDeps, run: _Run) -> Optional[dict]:
    try:
        status, atlas = await deps.retro.call('GET', f'sessions/{run.session_id}/atlas')
    except Exception:  # noqa: BLE001
        return None
    return atlas if status == 200 and isinstance(atlas, dict) else None


async def _call(deps: LeisureDeps, what: str, method: str, path: str, body: Optional[dict] = None):
    try:
        return await deps.retro.call(method, path, body)
    except Exception as exc:  # noqa: BLE001 -- best effort at the end; the journal names it
        logger.warning('retro_leisure_%s_failed', what, exc_info=True)
        return 0, {'detail': type(exc).__name__}


async def _finish(deps: LeisureDeps, run: _Run, reason: str, *, mode: str,
                  ledger: Optional[list], started: bool = True) -> RetroSessionEnded:
    save_id = save_error = None
    human = False
    if run.session_id:
        ok, since = await deps.store.seconds_since_human()
        human = (not ok) or (since is not None and since <= deps.clock() - run.started_at)
        if mode in (NORMAL, RELEASE) and run.epoch is not None:
            await _call(deps, 'policy_stop', 'POST', f'sessions/{run.session_id}/scripted', {
                'leisure_id': run.leisure_id, 'generation': run.generation, 'port': run.port,
                'action': 'stop', 'duration_seconds': 5, 'cognition': True, 'expected_epoch': run.epoch})
        if started and run.epoch is not None:
            if human:
                save_error = 'skipped:human_may_be_in_control'
            elif reason != 'retro_session_ended':
                status, saved = await _call(deps, 'save', 'POST', f'sessions/{run.session_id}/save', {})
                if status == 200 and isinstance(saved, dict):
                    save_id = (saved.get('saved') or {}).get('save_id')
                else:
                    save_error = str((saved or {}).get('detail') if isinstance(saved, dict) else status)
        if mode == NORMAL and not human:
            await _call(deps, 'session_stop', 'POST', f'sessions/{run.session_id}/stop', {})
    ended_at = deps.clock()
    if ledger is not None:
        rows = [dict(r, ended_at=ended_at, seconds=round(ended_at - run.started_at, 1))
                if r.get('leisure_id') == run.leisure_id else r for r in ledger]
        try:
            await deps.store.write_ledger(rows)
        except Exception:  # noqa: BLE001 -- the open row charges the full cap; fail safe
            logger.warning('retro_leisure_ledger_close_failed', exc_info=True)
    ended = RetroSessionEnded(
        session_id=run.session_id or '', lineage_id=run.lineage_id, leisure_id=run.leisure_id,
        game_id=run.game_id, started_at=run.started_at, ended_at=ended_at, end_reason=reason,
        act_id=run.act_id, loaded_save_id=run.loaded_save_id, end_save_id=save_id,
        end_save_error=save_error, decisions_planned=run.planned, planner_rejections=run.rejections,
        sheds=run.sheds, verdicts=dict(run.verdicts), human_may_be_in_control=human)
    await _journal(deps, 'ended' if started else 'refused', reason=reason, mode=mode, **{
        k: v for k, v in ended.as_dict().items() if k not in ('end_reason',)})
    await _status(deps, run, 'ended', end_reason=reason, end_save_id=save_id)
    return ended

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

# One source of truth for the context-seed char cap (review finding: was
# independently hardcoded here AND in store.py, two 240s to keep in sync by
# hand). store.py's top-level imports are stdlib-only -- importing it here
# costs nothing and creates no cycle (store.py never imports this module).
from .store import MAX_MEMORY_CRYSTALLIZATION_CONTEXT_CHARS as _MAX_MEMORY_CRYSTALLIZATION_CONTEXT_CHARS
from .store import MAX_REVERIE_CONTEXT_CHARS as _MAX_REVERIE_CONTEXT_CHARS
from .store import MAX_SELF_STUDY_CONTEXT_CHARS as _MAX_SELF_STUDY_CONTEXT_CHARS

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("orion-thought.settings")

# orion-mind runs 3 sequential LLM phases (semantic → appraisal → stance), each
# capped by MIND_LLM_TIMEOUT_SEC (default 60s on the orion-mind service). A wall
# budget below ~3× that ceiling guarantees synthesis is cut off mid-pipeline and
# the Mind degrades to contract_only (the empty-shell cognition failure mode).
# See fix/mind-enrichment-wall-budget.
MIND_LLM_TIMEOUT_SEC_ASSUMED: float = 60.0
MIND_ENRICHMENT_PHASE_COUNT: int = 3
MIND_ENRICHMENT_MIN_VIABLE_WALL_MS: int = int(
    MIND_LLM_TIMEOUT_SEC_ASSUMED * MIND_ENRICHMENT_PHASE_COUNT * 1000
)


class ThoughtSettings(BaseSettings):
    service_name: str = Field("orion-thought", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(7155, alias="THOUGHT_PORT")

    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    # RPC-health snapshot publish (orion:rpc_health:snapshot) + per-hop channel_latency
    # (A0 of docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md).
    # Defaults match .env_example. channel_latency is extra="forbid" on the consumer side:
    # rebuild orion-signal-gateway + orion-equilibrium-service on PR #2312's build first.
    rpc_health_publish_enabled: bool = Field(True, alias="RPC_HEALTH_PUBLISH_ENABLED")
    rpc_health_publish_interval_sec: float = Field(30.0, gt=0.0, alias="RPC_HEALTH_PUBLISH_INTERVAL_SEC")
    rpc_health_channel_latency_enabled: bool = Field(True, alias="RPC_HEALTH_CHANNEL_LATENCY_ENABLED")

    channel_thought_request: str = Field(
        "orion:thought:request",
        alias="CHANNEL_THOUGHT_REQUEST",
    )
    channel_thought_artifact: str = Field(
        "orion:thought:artifact",
        alias="CHANNEL_THOUGHT_ARTIFACT",
    )
    channel_thought_result_prefix: str = Field(
        "orion:thought:result:",
        alias="CHANNEL_THOUGHT_RESULT_PREFIX",
    )
    channel_cortex_exec_request: str = Field(
        "orion:cortex:exec:request",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_REQUEST", "CORTEX_EXEC_REQUEST_CHANNEL"),
        alias="CHANNEL_CORTEX_EXEC_REQUEST",
    )
    channel_cortex_exec_result_prefix: str = Field(
        "orion:exec:result",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_RESULT_PREFIX", "CORTEX_EXEC_RESULT_PREFIX"),
        alias="CHANNEL_CORTEX_EXEC_RESULT_PREFIX",
    )
    # 360, not 120 (2026-08-26). A shorter default is NOT the safe direction for
    # this key: an under-budgeted stance does not fail closed, it reports
    # `turn_deferred` -- which reads as "Orion judged the moment wrong" and is
    # indistinguishable from it at the Hub layer. Measured live: a stance that
    # completed correctly at 122s was thrown away at 120.006s. Must stay under
    # Hub's own TIMEOUT_SEC=400 outer wait. See services/orion-thought/.env_example.
    stance_react_timeout_sec: float = Field(360.0, alias="STANCE_REACT_TIMEOUT_SEC")
    # Unified-turn latency L4 (2026-10-06): send cortex-exec a
    # stance_context_prepare at the same time as the orion-mind call, so the
    # ~9 s stance context build overlaps mind instead of following it. Off ->
    # stance_react builds its context after mind, as before.
    stance_prepare_parallel_enabled: bool = Field(True, alias="ORION_THOUGHT_STANCE_PREPARE_PARALLEL")


    # --- Reveries that ask (docs/superpowers/specs/2026-11-18-reverie-questions-design.md, Patch 1) ---
    # A reverie starts only in a quiet stretch (arousal idle AND no attention winner for
    # QUIET_MIN_CALM_SEC), from one piece of real material (a seed), and leaves one question for
    # curiosity, or a reason there is none. The worker polls the gate every INTERVAL_SEC. ON, as shipped.
    reverie_enabled: bool = Field(True, alias="ORION_REVERIE_ENABLED")
    reverie_interval_sec: float = Field(60.0, gt=0.0, alias="ORION_REVERIE_INTERVAL_SEC")
    reverie_quiet_min_calm_sec: float = Field(600.0, ge=0.0, alias="ORION_REVERIE_QUIET_MIN_CALM_SEC")
    # Rolling 24 h. 0 = reverie writes nothing (the spec's off switch).
    reverie_max_per_day: int = Field(8, ge=0, alias="ORION_REVERIE_MAX_PER_DAY")
    # Rolling 24 h cap on reverie questions sent to the curiosity focus slot. 0 = questions stay recorded
    # but never reach curiosity. Hub enforces the same key a second time (curiosity_focus_listener.py).
    reverie_focus_max_per_day: int = Field(3, ge=0, alias="ORION_REVERIE_FOCUS_MAX_PER_DAY")
    # Centered-cosine ranks a far memory is taken from, "from:to" inclusive (Patch 0: 5:25).
    reverie_far_memory_band: str = Field("5:25", alias="ORION_REVERIE_FAR_MEMORY_BAND")
    # A seed ref used by a reverie in the last N days is spent; two questions on one thread_key inside
    # this window share a thread_id.
    reverie_thread_lapse_days: float = Field(14.0, gt=0.0, alias="ORION_REVERIE_THREAD_LAPSE_DAYS")
    # Patch 3 (app/reverie_threads.py): a question turned away only because the focus slot was full waits
    # and is offered again when the slot is free (no model call); the dig's outcome is read back onto the
    # question; a dig that said continue makes its thread the next reverie's seed. Off = Patch 1 behavior.
    # Requires services/orion-sql-db/manual_migration_reverie_question_threads.sql.
    reverie_threads_enabled: bool = Field(True, alias="ORION_REVERIE_THREADS_ENABLED")
    # Chroma that holds the orion_lived_memory index (orion-recall writes it; same URL recall reads).
    reverie_far_memory_chroma_url: str = Field(
        "http://orion-athena-vector-db:8000", alias="ORION_REVERIE_FAR_MEMORY_CHROMA_URL"
    )
    # Local time for the absence seed's window words ("between 07:00 and 08:00").
    reverie_timezone: str = Field("America/Denver", alias="ORION_REVERIE_TIMEZONE")
    regulation_reader_max_age_sec: float = Field(
        360.0, gt=0.0, alias="ORION_REGULATION_READER_MAX_AGE_SEC"
    )

    @property
    def reverie_far_memory_band_ranks(self) -> tuple[int, int]:
        lo, _, hi = str(self.reverie_far_memory_band or "5:25").partition(":")
        try:
            a, b = int(lo), int(hi)
        except ValueError:
            return (5, 25)
        a = max(1, a)
        return (a, max(a, b))


    notify_base_url: str = Field("http://orion-athena-notify:7140", alias="NOTIFY_BASE_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")

    # --- Reverie VISUAL chain (Patch 2 of docs/superpowers/specs/2026-08-20-
    # reverie-visual-chain-design.md, default-off). A second, parallel reverie
    # chain: generate an image via orion-diffusion-host, re-observe it via
    # orion-vision-host's existing caption_frame task, persist both, and
    # carry the caption forward as prior_description -- the enforced
    # continuity column design doc §2 built specifically to avoid repeating
    # the text chain's dead next_focus/drift fields. Independent worker loop
    # from reverie_chain above; sequential single-flight by construction (see
    # visual_chain.py module docstring), not a check-and-set marker.
    visual_chain_enabled: bool = Field(False, alias="ORION_VISUAL_CHAIN_ENABLED")
    # Act ledger (rung 2, "doing on one track"): each durable painting run publishes a
    # ``make`` act (act_opened on prepare, act_closed on produced/terminal/abandon) to
    # orion:grammar:event; orion-sql-writer persists and grades it into L10.
    thought_act_ledger_enabled: bool = Field(True, alias="THOUGHT_ACT_LEDGER_ENABLED")
    # Body economy Patch C3b: the painting / dream-hop generate step places a SHADOW bid
    # (orion.physiology.client -> durable-runs POST /physiology/bid) and logs it beside the step's
    # real outcome (app/physiology_shadow.py). Never obeyed; off = no bid, no log line.
    thought_physiology_shadow_bid_enabled: bool = Field(True, alias="THOUGHT_PHYSIOLOGY_SHADOW_BID_ENABLED")
    # Ladder retirement PR 2 (2026-11-05): Orion starts its own waking paintings
    # (app/painting_self_start.py): the visual-baseline schedule (config/reverie/visual_baseline.v1.yaml,
    # 5400 s after each painting, 600 s retry), arousal deferral, the self-monitor act gate, a SHADOW
    # hunger bid on thought.painting, then one reverie.visual durable run (POST <url>/runs). The
    # proposal template render_scene and proposal-runtime's baseline read are gone: this is the only
    # trigger. Off = Orion paints only when a dream hop or a manual run asks.
    painting_self_start_enabled: bool = Field(True, alias="ORION_PAINTING_SELF_START_ENABLED")
    painting_self_start_tick_sec: float = Field(60.0, gt=0.0, alias="ORION_PAINTING_SELF_START_TICK_SEC")
    painting_self_start_durable_runs_url: str = Field(
        "http://orion-athena-durable-runs:8121", alias="ORION_PAINTING_SELF_START_DURABLE_RUNS_URL"
    )
    # Temporal Self rev 4, R3 reader 4 (moved from orion-proposal-runtime with the schedule, same
    # name): while arousal reads `strained`, a due baseline painting is deferred by retry_sec.
    visual_baseline_arousal_enabled: bool = Field(True, alias="ORION_VISUAL_BASELINE_AROUSAL_ENABLED")
    # Self-monitor E1 (orion/self_monitor/gate.py), the gate execution-dispatch held for this act:
    # a painting faculty held broken/unreliable is refused `track_record_failing`, except one
    # re-test probe per window. Same names and defaults as orion-execution-dispatch-runtime.
    self_monitor_act_gate_enabled: bool = Field(True, alias="SELF_MONITOR_ACT_GATE_ENABLED")
    self_monitor_probe_interval_sec: float = Field(21600.0, ge=0.0, alias="SELF_MONITOR_PROBE_INTERVAL_SEC")
    # Slower than the text chain's ~90s cadence (design doc §4: "slower cadence
    # than the text chain"). Real cadence is max(this, actual run duration) --
    # intentional, not a bug (design doc §4).
    visual_chain_interval_sec: float = Field(600.0, alias="ORION_VISUAL_CHAIN_INTERVAL_SEC")
    # orion-diffusion-host on Circe (services/orion-diffusion-host README --
    # HOST_PORT=8014, same tailnet address the LLM gateway's route table uses
    # for circe-worker-agent-1, the llama.cpp worker this port used to serve).
    diffusion_host_base_url: str = Field(
        "http://100.112.254.99:8014", alias="ORION_DIFFUSION_HOST_BASE_URL"
    )

    # GPU2 MUTEX WITH orion-world-model, via orion-gpu-pool (stage 5.4; replaced the durable-runs
    # /capacity permit). The two share circe's gpu2 with no OS-level arbitration (two "CUDA-capable
    # device(s) is/are busy or unavailable" failures, 2026-09-24).
    # - The durable reverie-visual run already holds a pool `diffusion` hold; the generate step
    #   validates it, then ATTACHES a child lease under it (runs in the hold's slot, jumps its
    #   queue: no second wait). The child lives as long as the diffusion thread, so world stays off
    #   gpu2 even if the run gives the hold back mid-generate.
    # - A generate outside a durable run (the /visual-chain/run-once route, the legacy worker) has no
    #   hold, so it takes a plain `diffusion` request lease and waits at most this long.
    # The pool never places a `world` lease while either is active (gpu_pool.yaml serialize_with).
    # 90 s, not the old permit's 180: run-once's whole-run deadline (ORION_VISUAL_CHAIN_RUN_DEADLINE_SEC,
    # 300) must fit this wait + diffusion (120) + prompt/caption, or a long queue ends as a run-deadline
    # abandon instead of a clean resource_deferred. A deferral is a normal outcome; world-model's own
    # 2 s deadline still gives diffusion practical precedence on its native card.
    visual_chain_gpu_lease_deadline_sec: float = Field(
        90.0, alias="ORION_VISUAL_CHAIN_GPU_LEASE_DEADLINE_SEC"
    )

    # AMBIENT THERMAL GATE. GPU work heats the room Juniper sits in, and this is
    # the only budget here whose referent is outside Orion (see
    # orion/autonomy/thermal_gate.py). The reading comes from the cabinet
    # BME680 via orion-hub, which is on the same host as this worker -- so the
    # gate lives at the REQUESTER, not the GPU server: declining to ask is
    # cheaper and more honest than asking and being refused.
    thermal_gate_enabled: bool = Field(True, alias="ORION_THERMAL_GATE_ENABLED")
    # orion-hub runs with network_mode: host; THIS worker is bridge-networked, so
    # 127.0.0.1 is the container's own loopback and connection-refuses. Verified
    # live 2026-08-30 -- the gate failed open and reported degraded, which is the
    # designed behaviour, but it read nothing. The tailnet address is the same
    # convention AGENTS.md already mandates for ORION_BUS_URL. The Docker service
    # name does NOT resolve (hub is not on this network).
    cabinet_sensors_base_url: str = Field(
        "http://100.92.216.81:8080", alias="ORION_CABINET_SENSORS_BASE_URL"
    )
    cabinet_sensors_timeout_sec: float = Field(
        3.0, alias="ORION_CABINET_SENSORS_TIMEOUT_SEC"
    )
    # Trip and re-arm lines are NOT settings: they are owned by
    # orion/autonomy/thermal_gate.py (DEFAULT_HOT_C / DEFAULT_HOT_REARM_C) and read
    # from there by visual_chain.py (body economy Patch B: one owner for the hot line).
    # 30s (this field's original value) was tuned for sdxl-turbo's
    # single-step, near-instant generation. Real bug, caught live
    # 2026-08-28 the same day orion-diffusion-host swapped to
    # FLUX.1-schnell (design doc §19): FLUX's real 4-step generation with
    # CPU offloading measured 49-56s on the actual deployed hardware
    # (Circe, physical GPU 2) -- every visual-chain tick timed out and
    # recorded terminal_reason="generation_failed" the FIRST tick after
    # deploy that wasn't also caught by a manual-testing 429 collision.
    # 120s gives real margin over the observed 49-56s range (prompt-length
    # and shared-GPU-contention variance could push it higher) without
    # coming anywhere close to conflicting with visual_chain_interval_sec's
    # 600s tick cadence above.
    visual_chain_diffusion_timeout_sec: float = Field(
        120.0, alias="ORION_VISUAL_CHAIN_DIFFUSION_TIMEOUT_SEC"
    )
    # Total deadline for ONE whole visual-chain run, enforced around the
    # single-flight lock (visual_chain.py::run_visual_chain_once). NOT redundant
    # with the per-hop timeouts above: those sum rather than bound, and a
    # urllib socket timeout is reset by every received chunk, so no combination
    # of them caps how long the lock can be held. Sized above that sum --
    # interpretation 30 + diffusion 120 + percept upload 10 + caption 60 = 220s
    # of hop budget, plus the DB round trips -- so it only ever fires on a
    # genuine hang, never on a slow-but-working run.
    #
    # Deliberately LONGER than the caller's timeout (orion-cortex-exec's
    # thought_http_timeout_sec, 150s). A caller giving up must not abandon a run
    # mid-generation: the run finishes, persists its chain row and stores its
    # artifact, and only the dispatch reports a timeout. Losing a real image to
    # save a caller 70s of waiting is the wrong trade.
    visual_chain_run_deadline_sec: float = Field(
        300.0, alias="ORION_VISUAL_CHAIN_RUN_DEADLINE_SEC"
    )
    # Deadline for ONE durable `reverie.visual` generate step (visual_steps.py):
    # GPU2 capacity permit wait + diffusion + disk write. Per step, not the whole-run
    # deadline above. Never effectively below capacity budget + diffusion timeout +
    # 10s (visual_step_generate_deadline_sec() floors it), and should stay under the
    # run brief's per-step RPC budget (ReverieVisualRunBriefV1.timeout_sec, 360s) so
    # the reply lands before durable-runs gives up waiting. A generate still
    # recorded "generating" within 2x this window blocks a second diffusion call.
    visual_chain_step_generate_deadline_sec: float = Field(
        330.0, alias="ORION_VISUAL_CHAIN_STEP_GENERATE_DEADLINE_SEC"
    )
    # Backstop for an `active`/`unknown` reverie_visual_attempt nothing ever closed
    # (lost abandon, process death mid-run): every claim -- durable prepare and legacy
    # run-once alike -- releases one whose claim started longer ago than this
    # (result reason `attempt_expired`), unless a production receipt reconciles it to
    # produced. Must stay above the durable run's 5400s retry window plus margin; a
    # legacy run-once attempt lives minutes. A generate still inside its in-flight
    # window is never released by this.
    visual_chain_attempt_max_age_sec: float = Field(
        7200.0, alias="ORION_VISUAL_CHAIN_ATTEMPT_MAX_AGE_SEC", gt=0
    )
    # Watchdog for the failure mode visual_chain_run_deadline_sec above cannot
    # catch: the worker's own asyncio task wedged before ever reaching the
    # single-flight lock (confirmed live 2026-09-04, 24+ hours silent, zero
    # errors). ~4.5x the 600s tick cadence (visual_chain_interval_sec above) --
    # enough slack for a couple of legitimate thermal_refused cycles in a row
    # without false-alarming, still catches a real wedge well under an hour in.
    visual_chain_staleness_threshold_min: float = Field(
        45.0, alias="ORION_VISUAL_CHAIN_STALENESS_THRESHOLD_MIN", gt=0
    )
    # Painting-gap check (2026-10-10): hours since the last PRODUCED painting
    # (production receipt). Own loop started from main.py, NOT gated on
    # visual_chain_enabled (legacy worker flag, off in production -- which is
    # why the staleness watchdog above never ran during the 27.6 h 10-09/10
    # outage). Fires orion-notify severity "error". Threshold picked in-sample
    # on 2026-09-14..10-10: no gap between 7.6 h and 14.7 h, see
    # evals/test_painting_gap_replay.py.
    visual_painting_gap_check_enabled: bool = Field(
        True, alias="ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED"
    )
    visual_painting_gap_threshold_hours: float = Field(
        12.0, alias="ORION_VISUAL_PAINTING_GAP_THRESHOLD_HOURS", gt=0
    )
    visual_painting_gap_check_interval_sec: float = Field(
        600.0, alias="ORION_VISUAL_PAINTING_GAP_CHECK_INTERVAL_SEC", gt=0
    )
    # Failed-attempt streak (2026-10-20, same loop as the gap check): N painting attempts in a
    # row (reverie_visual_attempt) produced nothing -> CRITICAL card (emailed at once). 6 = one
    # more than the longest ordinary streak 2026-09-14..10-19; every streak >= 6 was an outage
    # (35 on 10-16, 16 on 10-09, 9 on 10-08 and 09-25). See evals/test_attempt_streak_replay.py.
    # Runs inside the painting-gap loop: ORION_VISUAL_PAINTING_GAP_CHECK_ENABLED=false stops it too.
    visual_attempt_streak_check_enabled: bool = Field(
        True, alias="ORION_VISUAL_ATTEMPT_STREAK_CHECK_ENABLED"
    )
    visual_attempt_failure_streak_threshold: int = Field(
        6, alias="ORION_VISUAL_ATTEMPT_FAILURE_STREAK_THRESHOLD", ge=1
    )
    # How often the watchdog re-checks reverie_visual_chain's newest row age.
    # 600s (not field-digester's own 900s precedent -- review finding, fixed:
    # that comment previously named 900s but this field's default was always
    # 600) matches visual_chain_interval_sec's own tick cadence directly above
    # -- cheap single query, no reason to poll faster than the thing it watches.
    visual_chain_watchdog_check_interval_sec: float = Field(
        600.0, alias="ORION_VISUAL_CHAIN_WATCHDOG_CHECK_INTERVAL_SEC", gt=0
    )
    # Patch 8 (visual_chain.py module docstring, design doc §22): the LLM interpretation step
    # between the selected context-seed and the diffusion prompt. Default ON -- this is the
    # actual fix for "how does this translate into fluffy cloud??", not an experiment to
    # observe unmitigated first (unlike reverie_metacog_background_enabled above, a routing
    # change with its own explicit observe-first request). Always plain `metacog` (never
    # `metacog_background`) -- see that call's own docstring for why it must never wait.
    visual_chain_interpretation_enabled: bool = Field(
        True, alias="ORION_VISUAL_CHAIN_INTERPRETATION_ENABLED"
    )
    # Live-measured 2026-08-29 (docs/superpowers/specs/2026-08-20-reverie-visual-chain-
    # design.md §21): metacog's real completions run ~1.7-2.3s, one observed 8.9s outlier.
    # 30s gives real margin without meaningfully delaying the diffusion call that follows.
    visual_chain_interpretation_timeout_sec: float = Field(
        30.0, alias="ORION_VISUAL_CHAIN_INTERPRETATION_TIMEOUT_SEC"
    )
    # Content-addressed image storage (orion.reverie.visual_storage, design
    # doc §6). Overridable so tests never touch the real mount.
    visual_chain_storage_dir: str = Field(
        "/mnt/storage-lukewarm/orion/reverie-visual",
        alias="ORION_VISUAL_CHAIN_STORAGE_DIR",
    )
    # orion-percept-store: the cross-host hop that lets a generated image
    # (produced on circe) reach orion-vision-host's caption_frame task
    # (athena) without assuming a shared filesystem -- same mechanism
    # orion-vision-council's foveal probe and orion-vision-frame-router
    # already use (see runner.py::_load_image_from_percept_store). Same
    # literal container hostname orion-vision-host's own
    # VISION_PERCEPT_STORE_URL already resolves (both are athena-network
    # services), not the tailnet address.
    visual_chain_percept_store_url: str = Field(
        "http://orion-athena-percept-store:8000/percepts",
        alias="ORION_VISUAL_CHAIN_PERCEPT_STORE_URL",
    )
    visual_chain_percept_store_token: str | None = Field(
        None, alias="ORION_VISUAL_CHAIN_PERCEPT_STORE_TOKEN"
    )
    visual_chain_percept_upload_timeout_sec: float = Field(
        10.0, alias="ORION_VISUAL_CHAIN_PERCEPT_UPLOAD_TIMEOUT_SEC"
    )
    # Originally the shared orion:exec:request:VisionHostService channel
    # (design doc §3: caption_frame + percept_sha256 already captions any
    # image, no vision-host code change needed, only a new producer). Moved
    # to circe's dedicated Qwen2-VL lane (services/orion-vision-host/
    # docker-compose.circe-qwen.yml) 2026-08-26 after live-confirming
    # athena's shared BLIP-base instance cannot produce a caption that
    # clears sanitize_caption's quality bar for a generated (not
    # camera-captured) image -- 3/3 real ticks came back uncaptioned.
    # reply_to is built per-call as f"{prefix}:{corr_id}", matching
    # orion:vision:reply:*'s documented wildcard (orion/bus/channels.yaml).
    channel_vision_host_request: str = Field(
        "orion:exec:request:VisionHostService:circe-vl", alias="CHANNEL_VISION_HOST_REQUEST"
    )
    channel_vision_reply_prefix: str = Field(
        "orion:vision:reply", alias="CHANNEL_VISION_REPLY_PREFIX"
    )
    visual_chain_caption_timeout_sec: float = Field(
        60.0, alias="ORION_VISUAL_CHAIN_CAPTION_TIMEOUT_SEC"
    )
    # Patch 3 context-seed tunables (store.py::load_latest_reverie_
    # interpretation) -- were bare module constants/unbounded until now,
    # unlike every other tunable in this block. reverie_context_char_limit's
    # default is store.py's own MAX_REVERIE_CONTEXT_CHARS (imported below,
    # not a second hardcoded 240) -- one source of truth, one-directional
    # import (settings -> store). Safe: store.py never imports this settings
    # module (its own module docstring: "never the heavy orion.substrate
    # package this thin service does not ship"), and its own top-level
    # imports are stdlib-only -- nothing heavy runs just by importing it here.
    #
    # reverie_context_max_age_sec closes a real staleness gap: without it, a
    # stalled/disabled text-reverie worker (chain.py) leaves the same old
    # thought answering forever, woven into the diffusion prompt and shown
    # in the visual cockpit's own context_text field as "Orion is currently
    # thinking" long after it stopped being current. This does NOT bound the
    # Hub Reverie tab's separate Text sub-view (reverie_routes.py::
    # text_recent), which has no staleness filter of its own and is not
    # touched by this change -- an operator can still see an old thought
    # presented as the latest one there.
    #
    # 900s (not the felt_state_reader.py convention of 2x the text-reverie
    # chain's own ~90s tick = 180s, and not proposal-runtime's
    # load_recent_reverie_thought's 300s default -- both real, checked, not
    # reused here on purpose): those two consumers read on their OWN fast
    # cadence, so a tight window matched to the *producer's* tick makes
    # sense for them. This context-seed is read once per VISUAL chain run
    # (ORION_VISUAL_CHAIN_INTERVAL_SEC, default 600s) -- a 180s window would
    # reject a perfectly fresh thought on almost every single visual-chain
    # tick, since 600s > 180s. 900s = 1.5x this consumer's own poll interval,
    # the same "multiple of the consumer's cadence" convention felt_state_
    # reader.py already uses, just computed against the right cadence.
    #
    # gt=0 on both (review finding): without it, a 0 or negative
    # ORION_REVERIE_CONTEXT_MAX_AGE_SEC makes the SQL freshness clause
    # permanently unsatisfiable (silently degrading to "no context-seed,
    # ever" -- indistinguishable from the genuine no-data case), and a
    # negative ORION_REVERIE_CONTEXT_CHAR_LIMIT turns Python's negative-index
    # slicing into "keep everything except the last N chars" -- the exact
    # opposite of a cap. Both fail loud at settings load instead of silently
    # doing the wrong thing at read time.
    reverie_context_char_limit: int = Field(
        _MAX_REVERIE_CONTEXT_CHARS, alias="ORION_REVERIE_CONTEXT_CHAR_LIMIT", gt=0
    )
    # 2026-11-24: the seed is now the newest accepted reverie question (at most 8 a day, only in quiet
    # stretches), so the window is a day, not the narration chain's 15 minutes.
    reverie_context_max_age_sec: float = Field(
        86400.0, alias="ORION_REVERIE_CONTEXT_MAX_AGE_SEC", gt=0
    )
    # Patch 4 (design doc §15): live 2026-08-27, `prior_description`
    # continuity can lock onto one visual attractor indefinitely (confirmed
    # live -- "ancient Roman aqueduct" imagery, unbroken across 10+ runs /
    # 100+ minutes, predating Patch 3's context-seeding and un-moved by it:
    # a short abstract context clause has nowhere near the prompt weight of
    # a long, concrete continuity description). After this many CONSECUTIVE
    # runs carrying continuity forward, the next run forces one reset --
    # drops prior_description from that run's prompt only (re-seeding from
    # context_text, or the fixed seed if neither exists) -- then continuity
    # resumes normally. Not disabled by setting this high without limit:
    # 0 would mean "reset every single run", never let continuity build
    # at all; there is no off switch by design, since an unbounded
    # continuity streak is exactly the failure mode this exists to bound.
    visual_chain_continuity_max_runs: int = Field(
        3, alias="ORION_VISUAL_CHAIN_CONTINUITY_MAX_RUNS"
    )

    # Source repointed 2026-11-07 (ladder retirement PR 6): curiosity's self-inquiry
    # answers (self_concept_history, ~6 a day; a 6 h window holds one 78.7% of the
    # time, read live), because the analysis rows below had one trigger, the retired
    # dispatch rung. The history that follows is about the old source.
    # Patch 5 (design doc §16): a second, richer context-seed alongside
    # reverie_context_* above -- self_study_analysis.py's four deterministic
    # window-contrast analyses (concept induction, vision events, affective
    # state, co-creation signals), real quantified self-observation rather
    # than a bare narration sentence. store.py's
    # `_SAFE_SELF_STUDY_SOURCE_PREFIXES` is the actual privacy boundary (an
    # allowlist of the four safe producers, not "any source_kind='self_study'
    # row") -- these two settings only tune length/freshness of content
    # already gated safe by that allowlist, same division of concerns
    # reverie_context_char_limit/max_age_sec has with load_latest_reverie_
    # interpretation's own EXISTS/hollow gate.
    #
    # 21600s (6h), not reverie_context_max_age_sec's 900s: these analyses
    # fire on their OWN 6-72h window-contrast cadence (real values seen in
    # bodies: "the last 6h against the 6h before it", "last 12h", "last
    # 72h") -- a 900s window would read as permanently absent almost every
    # tick. 6h is this producer's own shortest real window, the same
    # "match the producer's cadence, not the consumer's" reasoning
    # reverie_context_max_age_sec's own comment already applies elsewhere.
    #
    # gt=0 on both, same reasoning as reverie_context_char_limit/
    # max_age_sec's own comment: a non-positive value silently produces the
    # wrong behavior (permanently-unsatisfiable freshness clause / negative-
    # index slicing) instead of failing loud at settings load.
    self_study_context_char_limit: int = Field(
        _MAX_SELF_STUDY_CONTEXT_CHARS, alias="ORION_SELF_STUDY_CONTEXT_CHAR_LIMIT", gt=0
    )
    self_study_context_max_age_sec: float = Field(
        21600.0, alias="ORION_SELF_STUDY_CONTEXT_MAX_AGE_SEC", gt=0
    )
    # Stage 2 PR G: visual reverie reads the validated episode-memory store by
    # default. ``crystallization`` is a bounded rollback while the legacy
    # reader remains present; Stage 2 PR H removes that reader at cutover.
    visual_memory_seed_source: Literal["episode_memory", "crystallization"] = Field(
        "episode_memory", alias="REVERIE_VISUAL_MEMORY_SEED_SOURCE"
    )
    memory_crystallization_context_char_limit: int = Field(
        _MAX_MEMORY_CRYSTALLIZATION_CONTEXT_CHARS,
        alias="ORION_MEMORY_CRYSTALLIZATION_CONTEXT_CHAR_LIMIT",
        gt=0,
    )
    # 604800s (7 days), NOT self_study_context_max_age_sec's 6h. Stage 2
    # episode-memory eligibility is explicitly bounded to seven days; the
    # same field remains wired to the legacy crystallization reader during
    # rollback so this cutover does not add a second drifting age control.
    # The episode reader clamps this value to seven days, so an operator may
    # narrow but cannot widen its governance boundary.
    #
    # 7 days, not this fix's first draft of 3 days: a code-review pass on
    # this exact patch caught that the 3-day pick rested on an all-time-max
    # query silently narrowed to "last 14 days" without saying so or
    # reconciling it against the earlier (unscoped) PR #1917 write-up,
    # which had found a >10-day gap. Re-querying the FULL history
    # (2026-08-28) confirms both numbers were real, not contradictory:
    # ALL 5 of the largest gaps ever observed are old --the two biggest are
    # 10d5h (2026-07-31 -> 08-11) and 3d10h (2026-07-25 -> 07-29), both from
    # more than 2 weeks before this patch. Every gap since 2026-08-11 (17+
    # days of real recent activity) has stayed under 2 days. 7 days covers
    # the recent pattern with real margin and covers the second-largest
    # historical gap too; it does NOT cover the single 10-day outlier from
    # early August. That is an accepted, explicit tradeoff, not an
    # oversight: a dry spell that long would read as absent again, which is
    # the same honest degrade-to-absent behavior every context-seed reader
    # in this file has -- unlike the 6h bug this patch fixes, a 7-day gap
    # is a genuinely rare, real quiet period, not the every-tick norm.
    memory_crystallization_context_max_age_sec: float = Field(
        604800.0, alias="ORION_MEMORY_CRYSTALLIZATION_CONTEXT_MAX_AGE_SEC", gt=0
    )

    # --- Mind stance enrichment (unified turn; default-off) ---
    # Runs orion-mind before stance_react and injects an advisory self/attention
    # coloring. Silent no-op unless orion-mind has MIND_LLM_SYNTHESIS_ENABLED=true
    # (a separate service — not visible to this service's env-parity check).
    #
    # Wall/timeout budget: orion-mind runs THREE sequential LLM phases (semantic →
    # appraisal → stance), each capped by MIND_LLM_TIMEOUT_SEC (default 60s on the
    # orion-mind service). A wall below ~3× that ceiling cuts synthesis off
    # mid-pipeline and forces contract_only degradation, so the wall default must
    # stay >= MIND_ENRICHMENT_MIN_VIABLE_WALL_MS and the HTTP read timeout must
    # exceed the wall (so Mind's own fail-open result is returned, not aborted).
    mind_enrichment_enabled: bool = Field(False, alias="ORION_THOUGHT_MIND_ENRICHMENT_ENABLED")
    mind_base_url: str = Field("http://mind:6611", alias="ORION_MIND_BASE_URL")
    mind_timeout_sec: float = Field(210.0, alias="ORION_THOUGHT_MIND_TIMEOUT_SEC")
    mind_wall_ms: int = Field(180_000, alias="ORION_THOUGHT_MIND_WALL_MS")
    mind_router_profile: str = Field("default", alias="ORION_THOUGHT_MIND_ROUTER_PROFILE")
    mind_max_response_bytes: int = Field(2_000_000, alias="ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES")
    mind_artifact_publish_enabled: bool = Field(
        False, alias="ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED"
    )
    mind_coloring_max_items: int = Field(3, alias="ORION_THOUGHT_MIND_COLORING_MAX_ITEMS")
    channel_mind_artifact: str = Field("orion:mind:artifact", alias="CHANNEL_MIND_ARTIFACT")

    # --- Temporal Self patch 4: "my day" cue for stance_react (app/temporal_self_cue.py) ---
    # One short paragraph rendered from orion-durable-runs' current day frame (no LLM), shown to
    # the unified turn's stance step. Kill switch: false = stance_react runs exactly as before.
    # The frame older than MAX_AGE_SEC, for another local day, or not fetched within
    # FETCH_TIMEOUT_SEC renders nothing. The fetch overlaps Mind enrichment.
    temporal_self_stance_cue_enabled: bool = Field(True, alias="TEMPORAL_SELF_STANCE_CUE_ENABLED")
    temporal_self_frame_url: str = Field("http://durable-runs:8121", alias="TEMPORAL_SELF_FRAME_URL")
    temporal_self_cue_max_age_sec: float = Field(900.0, gt=0.0, alias="TEMPORAL_SELF_CUE_MAX_AGE_SEC")
    temporal_self_cue_fetch_timeout_sec: float = Field(
        0.5, gt=0.0, le=5.0, alias="TEMPORAL_SELF_CUE_FETCH_TIMEOUT_SEC"
    )

    # --- Personality graph read-back (app/personality_cue.py; spec 2026-10-16 rev 2) ---
    # What Orion keeps coming back to, in Orion's own words, rendered by a fixed template
    # (orion/personality/cue.py, no LLM) from orion-durable-runs' GET /personality/frame. A record,
    # not an instruction. Shown to stance_react (fetched alongside Mind) and reverie (alongside its
    # other reads). Missing / stale (> MAX_AGE_SEC) / empty frame or no reply within TIMEOUT_SEC
    # renders nothing. Kill switch: false = both prompts exactly as before.
    personality_cue_enabled: bool = Field(False, alias="PERSONALITY_CUE_ENABLED")
    personality_frame_url: str = Field("http://durable-runs:8121", alias="PERSONALITY_FRAME_URL")
    personality_cue_timeout_sec: float = Field(0.5, gt=0.0, le=5.0, alias="PERSONALITY_CUE_TIMEOUT_SEC")
    personality_cue_max_age_sec: float = Field(3600.0, gt=0.0, alias="PERSONALITY_CUE_MAX_AGE_SEC")
    personality_cue_top_k: int = Field(5, ge=1, le=12, alias="PERSONALITY_CUE_TOP_K")

    # --- Self-monitor track record (app/self_track_record_cue.py; spec 2026-10-20 step 3) ---
    # What Orion is failing at or misjudging right now (broken / unreliable / miscalibrated
    # faculties only, <= 300 chars), rendered by a fixed template (orion/self_monitor/cue.py, no
    # LLM) from the state orion-durable-runs publishes to Redis orion:self_monitor:latest on
    # ORION_BUS_URL. Read every 5 min in the background; absent / stale / unparseable renders
    # nothing. Shown to stance_react and reverie's coalition branch. false = both prompts as before.
    self_track_record_cue_enabled: bool = Field(True, alias="SELF_TRACK_RECORD_CUE_ENABLED")

    # --- Intention ledger D3: open promises for stance_react (app/open_promises_cue.py) ---
    # Up to 3 of Orion's open promises (hub.promise acts, oldest first) on Juniper's own turns. Ships
    # OFF on purpose: the promise acts are shadow until #3531 measures the finder's live precision,
    # and a false promise shown to stance would steer Orion. true = render the block.
    open_promises_cue_enabled: bool = Field(False, alias="THOUGHT_OPEN_PROMISES_CUE_ENABLED")

    # --- Self-readings: body signals against their own 7 days (self-calibration spec section C) ---
    # app/self_readings_cue.py -> orion/attention/self_readings.py. A fixed template (no LLM) over
    # substrate_node_prediction_error_history, calibrated by the attention seam's own rule; only
    # unusual signals render, a calm body renders nothing. Kill switches: false = that prompt
    # exactly as before. The read+render is bounded by FETCH_TIMEOUT_SEC (renders nothing on miss).
    # Unified-turn self-prediction (docs/superpowers/specs/2026-10-23-unified-turn-self-prediction-
    # design.md, A1): stance also writes a private bet about this turn's own behaviour
    # (ThoughtEventV1.self_prediction). Never shown to the motor or judge. false = the stance
    # prompt exactly as before.
    stance_self_prediction_enabled: bool = Field(True, alias="STANCE_SELF_PREDICTION_ENABLED")
    self_readings_stance_enabled: bool = Field(True, alias="SELF_READINGS_STANCE_ENABLED")
    self_readings_fetch_timeout_sec: float = Field(
        0.5, gt=0.0, le=5.0, alias="SELF_READINGS_FETCH_TIMEOUT_SEC"
    )
    # Continuous body signals must last (spec docs/superpowers/specs/2026-10-16-
    # attention-continuous-signal-quota-design.md, option (b)). A body node the
    # glossary marks as read every tick (none today: biometrics, bus_synaptic retired 2026-11) qualifies
    # only when hot for more than 300 s of the last 600 s of its own readings
    # and hot now (orion.attention.sustained). false = today's verdict exactly.
    # Ships on (Juniper's rule).
    attention_sustained_body_enabled: bool = Field(True, alias="ATTENTION_SUSTAINED_BODY_ENABLED")
    # What each body signal tends to lead to (self-calibration spec section E): a self-reading line
    # gets "; when this is high, my next hour had <outcome> Nx as often at equal activity (...) [assoc:<id>]" from the
    # newest self_signal_outcome_association row (orion-equilibrium-service's hourly rollup) only
    # when both sides have enough hours and the 95% interval excludes 1. Kill switch: false = the
    # self-readings block is byte-identical to before.
    self_signal_outcome_association_enabled: bool = Field(
        True, alias="SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED"
    )
    # Repeated looks: the association is re-tested every hour, so a clause renders only after its
    # verdict (same direction) held on each of the last CONSECUTIVE_HOLDS hourly computations; a
    # missing hour or a different verdict resets the run. Same rule and default as
    # orion-equilibrium-service's EQUILIBRIUM_SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS.
    self_signal_outcome_consecutive_holds: int = Field(
        6, ge=1, le=48, alias="SELF_SIGNAL_OUTCOME_CONSECUTIVE_HOLDS"
    )

    # --- Integration patch 3: reverie reads Orion's now (app/reverie.py build_reverie_now_cue) ---
    # Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md (A3/A4). Each reverie
    # reads GET /temporal-self/now?organ=reverie on TEMPORAL_SELF_FRAME_URL and posts the question's cites
    # to POST /temporal-self/now/use at its close (#3499). The now off on the server (404), unreachable, or
    # slower than FETCH_TIMEOUT_SEC -> the reverie runs on its seed alone, logged reverie_now_unavailable.
    # The same timeout bounds the cites post. Kill switch: false = the seed alone.
    reverie_now_enabled: bool = Field(True, alias="ORION_REVERIE_NOW_ENABLED")
    reverie_now_fetch_timeout_sec: float = Field(
        2.0, gt=0.0, le=5.0, alias="ORION_REVERIE_NOW_FETCH_TIMEOUT_SEC"
    )

    # --- Integration patch 7b: the painting reads Orion's now (app/visual_chain.py compute_visual_plan) ---
    # Before the interpret call (visual_context_interpret, the painting's one model step) a waking painting
    # reads GET /temporal-self/now?organ=painting on TEMPORAL_SELF_FRAME_URL, shows the items to that call,
    # and posts the [experience:<id>] tags it wrote to POST /temporal-self/now/use (the tags are stripped
    # before the diffusion prompt). No interpret call this run (no context slot, interpretation off) -> no
    # read. Dream hop pictures never read. Same fail-open rules and timeout shape as the reverie's read.
    # Kill switch: false = the painting as before (no read, no tags asked for).
    painting_now_enabled: bool = Field(True, alias="ORION_PAINTING_NOW_ENABLED")
    painting_now_fetch_timeout_sec: float = Field(
        2.0, gt=0.0, le=5.0, alias="ORION_PAINTING_NOW_FETCH_TIMEOUT_SEC"
    )

    # --- Reasoning activity projection (always-on consumer; harmless when idle) ---
    # Consume per-call ReasoningCallV1 telemetry and materialize a rolling-window
    # ReasoningActivityV1 for φ. The buffer is capped (max_calls) so memory is
    # bounded regardless of producer rate.
    channel_reasoning_call: str = Field(
        "orion:cognition:reasoning_call", alias="CHANNEL_REASONING_CALL"
    )
    reasoning_activity_window_sec: float = Field(
        120.0, alias="REASONING_ACTIVITY_WINDOW_SEC"
    )
    reasoning_activity_max_calls: int = Field(
        2000, alias="REASONING_ACTIVITY_MAX_CALLS"
    )


def mind_enrichment_config_warnings(s: "ThoughtSettings") -> list[str]:
    """Deterministic boot-time coherence checks for the Mind enrichment budget.

    Only meaningful when enrichment is enabled. Returns human-readable warnings
    for budget settings that would silently force contract_only degradation.
    """
    warnings: list[str] = []
    if not s.mind_enrichment_enabled:
        return warnings
    if s.mind_wall_ms < MIND_ENRICHMENT_MIN_VIABLE_WALL_MS:
        warnings.append(
            f"wall_too_small ORION_THOUGHT_MIND_WALL_MS={s.mind_wall_ms} < "
            f"min_viable={MIND_ENRICHMENT_MIN_VIABLE_WALL_MS}: 3-phase LLM synthesis "
            "will be cut off mid-pipeline and Mind will degrade to contract_only"
        )
    if s.mind_timeout_sec * 1000.0 <= s.mind_wall_ms:
        warnings.append(
            f"http_timeout_not_above_wall ORION_THOUGHT_MIND_TIMEOUT_SEC={s.mind_timeout_sec}s "
            f"(<= ORION_THOUGHT_MIND_WALL_MS={s.mind_wall_ms}ms): the HTTP client may abort "
            "before Mind returns its own fail-open result, losing diagnostics/artifact"
        )
    return warnings


settings = ThoughtSettings()
logger.info(
    "Loaded orion-thought settings service=%s v=%s port=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
)
for _mind_cfg_warning in mind_enrichment_config_warnings(settings):
    logger.warning("mind_enrichment_config %s", _mind_cfg_warning)

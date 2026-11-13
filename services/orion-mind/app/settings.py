from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _SERVICE_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_NAME: str = Field(default="orion-mind", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="unknown", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    PORT: int = Field(default=6611, alias="PORT")
    MIND_SNAPSHOT_MAX_BYTES: int = Field(default=512_000, alias="MIND_SNAPSHOT_MAX_BYTES")
    MIND_WALL_MS_DEFAULT: int = Field(default=180_000, alias="MIND_WALL_MS_DEFAULT")
    MIND_N_LOOPS_DEFAULT: int = Field(default=1, alias="MIND_N_LOOPS_DEFAULT")
    MIND_ROUTER_PROFILES_PATH: str = Field(
        default="",
        alias="MIND_ROUTER_PROFILES_PATH",
        description="Directory containing router_profiles.yaml; empty = app/config beside settings",
    )

    MIND_EVIDENCE_MAX_CHARS: int = Field(default=12_000, alias="MIND_EVIDENCE_MAX_CHARS")
    MIND_EVIDENCE_MAX_MESSAGES: int = Field(default=8, alias="MIND_EVIDENCE_MAX_MESSAGES")
    MIND_EVIDENCE_MAX_RECALL_FRAGMENTS: int = Field(default=8, alias="MIND_EVIDENCE_MAX_RECALL_FRAGMENTS")
    MIND_EVIDENCE_MAX_PROJECTION_ITEMS: int = Field(default=16, alias="MIND_EVIDENCE_MAX_PROJECTION_ITEMS")
    # RECALL_TRANSPORT_PG_DSN / RECALL_TRANSPORT_RENDER_GATE_THRESHOLD removed
    # 2026-11 with recall_signal_resolver.py: the bus_synaptic transport line
    # was gated on node:substrate.bus_synaptic, retired in PE suite rebuild
    # step 5.

    MIND_LLM_SYNTHESIS_ENABLED: bool = Field(default=False, alias="MIND_LLM_SYNTHESIS_ENABLED")
    # 2026-09-07: semantic moved off `quick` (circe 8013, saturated -- 8 inflight
    # + up to 11 queued, 30-115s waits, timing out Mind's 60s budget on ~97% of
    # live chat turns) onto `metacog` (circe 8012). stance's code default here
    # was `chat`, but the live-deployed .env had already overridden it to
    # `quick` (same saturated lane as semantic, same failure) -- so this moves
    # it off `quick` for the same reason, not away from `chat`. appraisal was
    # already on metacog, unchanged.
    #
    # metacog was confirmed idle (4/4 slots) via live /slots + zero real
    # gateway_upstream_queued/overloaded events at the moment of this patch --
    # NOT load-tested against 3 sequential Mind calls per turn plus whatever
    # metacog_background traffic is mid-wait. metacog is a real 4-slot shared
    # resource (memory-consolidation, vision-council, reverie, hub's concept
    # classifier, and orion-cortex-exec's daily metacog/journal draft all use
    # it too) and orion-llm-gateway/.env_example's own LLM_GATEWAY_UPSTREAM_
    # MAX_INFLIGHT comment documents a 2026-09-05 flood that starved metacog
    # calls for 19 min (a gateway thread-pool sizing issue since fixed by
    # configure_executor(), not proof metacog itself can't saturate -- but a
    # sign this lane has been contended before). If chat turns start showing
    # the same fallback_contract_only pattern this patch fixes, check
    # `docker logs orion-llm-gateway` for queued/overloaded against metacog's
    # upstream before assuming semantic_synthesis itself regressed.
    MIND_SEMANTIC_MODEL_ROUTE: str = Field(default="metacog", alias="MIND_SEMANTIC_MODEL_ROUTE")
    MIND_APPRAISAL_MODEL_ROUTE: str = Field(default="metacog", alias="MIND_APPRAISAL_MODEL_ROUTE")
    MIND_STANCE_MODEL_ROUTE: str = Field(default="metacog", alias="MIND_STANCE_MODEL_ROUTE")
    # On Juniper's live turn (engine.carries_live_turn: utterance_origin="juniper" only; Orion's
    # own and unmarked runs stay `system`), every phase configured as `metacog` uses this
    # route instead: same pool class, `interactive` priority, so a heat shed never refuses the
    # turn's Mind calls (spec 2026-10-06-thermal-controller-redesign D4). Empty = rollback (turns
    # use the phase routes above, `system`, and are shed with background work).
    MIND_TURN_MODEL_ROUTE: str = Field(default="metacog_turn", alias="MIND_TURN_MODEL_ROUTE")
    MIND_LLM_TIMEOUT_SEC: float = Field(default=60.0, alias="MIND_LLM_TIMEOUT_SEC")
    # Per-phase LLM cap for a held run (req.gpu_lease set: world pulse, curiosity, journal, field
    # notes), which runs on its own leased seat (agent / agent-gpu2). Bonsai-2-27B on agent-gpu2
    # thinks first: a 2048-token semantic reply at its slow end (~14 tok/s) needs ~146 s. The run's
    # wall still bounds the whole run.
    MIND_LLM_HELD_TIMEOUT_SEC: float = Field(default=150.0, gt=0.0, alias="MIND_LLM_HELD_TIMEOUT_SEC")
    MIND_LLM_MAX_TOKENS_SEMANTIC: int = Field(default=2048, alias="MIND_LLM_MAX_TOKENS_SEMANTIC")
    MIND_LLM_MAX_TOKENS_APPRAISAL: int = Field(default=3072, alias="MIND_LLM_MAX_TOKENS_APPRAISAL")
    MIND_LLM_MAX_TOKENS_STANCE: int = Field(default=1536, alias="MIND_LLM_MAX_TOKENS_STANCE")
    MIND_LLM_THINKING_APPRAISAL: bool = Field(default=True, alias="MIND_LLM_THINKING_APPRAISAL")
    # Judge thinking on a held run's seat (agent / agent-gpu2). Off: with thinking on, both seat models
    # spent the whole appraisal token budget reasoning and returned no answer (issue #3294). Unheld
    # runs keep MIND_LLM_THINKING_APPRAISAL.
    MIND_LLM_THINKING_APPRAISAL_HELD: bool = Field(default=False, alias="MIND_LLM_THINKING_APPRAISAL_HELD")
    MIND_LLM_FAIL_OPEN_LEGACY: bool = Field(default=True, alias="MIND_LLM_FAIL_OPEN_LEGACY")
    MIND_LLM_PHASE_SAFETY_MS: float = Field(default=50.0, alias="MIND_LLM_PHASE_SAFETY_MS")
    MIND_LLM_RETURN_LOGPROBS_SEMANTIC: bool = Field(default=False, alias="MIND_LLM_RETURN_LOGPROBS_SEMANTIC")
    MIND_LLM_LOGPROB_PROBE_MODE: str = Field(
        default="",
        alias="MIND_LLM_LOGPROB_PROBE_MODE",
        description="Empty=OpenAI chat logprobs; native_completion=aligned llama.cpp /completion (requires gateway LLM_LOGPROB_NATIVE_COMPLETION_ENABLED).",
    )
    MIND_LLM_UNCERTAINTY_METACOG_ENABLED: bool = Field(default=False, alias="MIND_LLM_UNCERTAINTY_METACOG_ENABLED")
    MIND_METACOG_TRIGGER_CHANNEL: str = Field(
        default="orion:equilibrium:metacog:trigger",
        alias="MIND_METACOG_TRIGGER_CHANNEL",
    )

    MIND_LLM_USE_BUS: bool = Field(default=True, alias="MIND_LLM_USE_BUS")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    # Repo-wide bus convention (see root CLAUDE.md): always the real tailscale node address,
    # never bus-core/localhost/the docker-compose service name. This default was stale
    # (redis://redis:6379/0) before this patch -- the checked-in .env_example already had the
    # correct value, only this code-level fallback was wrong.
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, alias="HEARTBEAT_INTERVAL_SEC")
    # Periodic RpcHealthSnapshotV1 publish to orion:rpc_health:snapshot (mesh transport
    # coverage, docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-
    # baseline-design.md). Per-call LLM-gateway buses fold into
    # app.llm_client.RPC_HEALTH_SINK; a long-lived lifespan bus publishes it.
    # Defaults match .env_example.
    RPC_HEALTH_PUBLISH_ENABLED: bool = Field(default=True, alias="RPC_HEALTH_PUBLISH_ENABLED")
    RPC_HEALTH_PUBLISH_INTERVAL_SEC: float = Field(default=30.0, gt=0.0, alias="RPC_HEALTH_PUBLISH_INTERVAL_SEC")
    RPC_HEALTH_CHANNEL_LATENCY_ENABLED: bool = Field(default=True, alias="RPC_HEALTH_CHANNEL_LATENCY_ENABLED")
    MIND_LLM_INTAKE_CHANNEL: str = Field(
        default="orion:exec:request:LLMGatewayService",
        alias="MIND_LLM_INTAKE_CHANNEL",
    )
    MIND_LLM_REPLY_PREFIX: str = Field(default="orion:mind:llm:reply", alias="MIND_LLM_REPLY_PREFIX")
    # Orion Prime 2A (docs/superpowers/specs/2026-10-29-orion-prime-design.md): the resident loop
    # (app/prime_loop.py). Off: no loop, no bids, no acts; run_mind is unaffected either way.
    PRIME_ENABLED: bool = Field(default=True, alias="PRIME_ENABLED")
    # The CPU tick: reads body + attention, records changes. Never calls a model.
    PRIME_TICK_SEC: float = Field(default=15.0, gt=0.0, alias="PRIME_TICK_SEC")
    # Least time between two thoughts. Price decides frequency above this floor.
    PRIME_REFRACTORY_SEC: float = Field(default=300.0, ge=0.0, alias="PRIME_REFRACTORY_SEC")
    # How often open questions are checked against curiosity's runs (one Hub read).
    PRIME_SETTLE_INTERVAL_SEC: float = Field(default=300.0, gt=0.0, alias="PRIME_SETTLE_INTERVAL_SEC")
    # circe chat lane 0 at pool priority system (config/gpu_pool.yaml route `prime`).
    PRIME_LLM_ROUTE: str = Field(default="prime", alias="PRIME_LLM_ROUTE")
    # The same call at pool priority `starved`, used only after a try that got no pool seat
    # (config/gpu_pool.yaml route `prime_starved`; orion/prime/state.py wants_starved_route).
    PRIME_STARVED_LLM_ROUTE: str = Field(default="prime_starved", alias="PRIME_STARVED_LLM_ROUTE")
    # How long one try may wait for a pool seat BEFORE its PRIME_LLM_TIMEOUT_SEC of generation
    # (llm-gateway option gateway_pool_wait_sec). 0 = no separate seat wait (the old single budget).
    PRIME_SEAT_WAIT_SEC: float = Field(default=600.0, ge=0.0, le=900.0, alias="PRIME_SEAT_WAIT_SEC")
    # The pool role the bid names (config/physiology/families.yaml family `prime`).
    PRIME_BID_LANE: str = Field(default="chat", alias="PRIME_BID_LANE")
    PRIME_MAX_TOKENS: int = Field(default=700, gt=0, alias="PRIME_MAX_TOKENS")
    PRIME_LLM_TIMEOUT_SEC: float = Field(default=60.0, gt=0.0, alias="PRIME_LLM_TIMEOUT_SEC")
    # Share of questions held back (logged as acts, never sent): the narrator control arm.
    PRIME_HOLDBACK_FRACTION: float = Field(default=0.2, ge=0.0, le=1.0, alias="PRIME_HOLDBACK_FRACTION")
    # Body (/physiology/state) and attention (/temporal-self/frame).
    PRIME_DURABLE_RUNS_URL: str = Field(default="http://orion-athena-durable-runs:8121", alias="PRIME_DURABLE_RUNS_URL")
    # Hub (host network on athena): curiosity's runs and pending focus (/curiosity/api/runs).
    PRIME_HUB_URL: str = Field(default="http://100.92.216.81:8080", alias="PRIME_HUB_URL")
    # Orion Prime step 3: Prime may send "slow" / "quicken" signals to a want family that declares the
    # receptor (config/physiology/families.yaml; today curiosity investigate), as bounded, temporary
    # charges in the body economy (orion/physiology/prime_charge.py). false: the move is not offered and a
    # regulate answer is refused; signals already sent are still graded.
    PRIME_REGULATE_ENABLED: bool = Field(default=True, alias="PRIME_REGULATE_ENABLED")
    # Where Prime reads the receptors: the read-only repo mount (docker-compose volumes).
    PRIME_FAMILIES_PATH: str = Field(default="/repo/config/physiology/families.yaml", alias="PRIME_FAMILIES_PATH")
    # Attention as the want competition, patch 2: Prime's concerns come from the events Orion's body economy
    # FUNDED (durable-runs GET /physiology/event-wants, from want_ledger: reading and camera-ask event wants)
    # instead of Temporal Self attention arcs. A new concern must name one of those events. false: arcs, as
    # before. ROLL BACK WITH THIS FLAG, not an image revert: an older image cannot read a saved concern that
    # carries event_ref and starts Prime empty. Shipped ON: in the 14-day replay 42 of 43 real concern bets had a
    # funded event to name (services/orion-durable-runs/evals/event_wants_replay.py; that eval exits 1 on a
    # separate check, camera salience is not mostly low; see the patch 2 PR report).
    PRIME_CONCERNS_FROM_EVENT_WANTS: bool = Field(default=True, alias="PRIME_CONCERNS_FROM_EVENT_WANTS")
    # Event wants decided in this many hours are considered; at most this many are shown (highest net first).
    PRIME_EVENT_WANTS_HOURS: float = Field(default=24.0, gt=0.0, le=72.0, alias="PRIME_EVENT_WANTS_HOURS")
    PRIME_EVENT_WANTS_LIMIT: int = Field(default=8, ge=1, le=50, alias="PRIME_EVENT_WANTS_LIMIT")
    # Living Prime Part 2, every turn ends back in Prime (orion/prime/handback.py, app/prime_loop.py): Prime listens
    # to the cockpit hops Hub already publishes (orion:cockpit:hop) and folds each finished chat, curiosity, consult
    # and letter turn into its state on the tick. A turn that made a promise opens a concern (at most 1 held, 4 a
    # day); a turn whose prompt cites an open prime_concern:<id> and replied settles it. false: no listener, no
    # fold, no frame section; concerns already opened from turns stay ordinary concerns. Shipped ON.
    PRIME_TURN_HANDBACK_ENABLED: bool = Field(default=True, alias="PRIME_TURN_HANDBACK_ENABLED")
    # Living Prime Part 3 (app/prime_loop.py, orion/prime/day_letter.py): the day letter is Prime's end-of-day
    # turn. When yesterday is closed and has no letter, Prime opens a `make` act and orders the letter run from Hub
    # (PRIME_HUB_URL). Tier 0: no bid. false = Prime never writes a letter (Hub's timer is retired: no letters).
    PRIME_DAY_LETTER_ENABLED: bool = Field(default=True, alias="PRIME_DAY_LETTER_ENABLED")
    # Runs ordered for one day at most: a run that wrote nothing is ordered again while the deadline holds.
    PRIME_DAY_LETTER_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=10, alias="PRIME_DAY_LETTER_MAX_ATTEMPTS")
    # The letter turn's budget per attempt (the run's timeout_sec).
    PRIME_DAY_LETTER_TURN_TIMEOUT_SEC: float = Field(default=1800.0, ge=60.0, le=3600.0,
                                                     alias="PRIME_DAY_LETTER_TURN_TIMEOUT_SEC")
    # The letter runs' shared admission deadline, counted from the act's open (every attempt must finish by it).
    PRIME_DAY_LETTER_RUN_DEADLINE_HOURS: float = Field(default=12.0, gt=0.0, le=48.0,
                                                       alias="PRIME_DAY_LETTER_RUN_DEADLINE_HOURS")
    # Intentions Prime takes on (Step -1, docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md;
    # orion/prime/intention.py, app/prime_loop.py): the take_on move on a concern whose source has an adapter (today:
    # a promise Orion made in chat). Prime opens a `make` act and orders ONE intention.carry run from Hub
    # (PRIME_HUB_URL); the reply goes back into the originating chat thread. false = no move, no frame section, no
    # order, nothing watched (runs already ordered finish or expire). Shipped ON.
    PRIME_INTENTION_TAKE_ON_ENABLED: bool = Field(default=True, alias="PRIME_INTENTION_TAKE_ON_ENABLED")
    # Orion's consent to a Claude tool call (docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-
    # design.md section 3; app/prime_consent.py): orion-mind answers orion:prime:consent:request with one short
    # model call (frame <= 4,000 chars) on route PRIME_CONSENT_LLM_ROUTE (circe chat lane 0, pool priority
    # interactive: Juniper waits on Claude's reply). Each answer that reached the model is an
    # act.prime.infer.consent act. false: no listener; the broker's asks find nobody and every "ask" is a no.
    # Shipped ON.
    PRIME_CONSENT_ENABLED: bool = Field(default=True, alias="PRIME_CONSENT_ENABLED")
    PRIME_CONSENT_LLM_ROUTE: str = Field(default="prime_consent", alias="PRIME_CONSENT_LLM_ROUTE")
    # Seat wait, then generation (the gateway's gateway_pool_wait_sec / gateway_read_timeout_sec). Their sum
    # plus the RPC margin stays under the broker's 40 s deadline (CLAUDE_TOOLS_CONSENT_DEADLINE_SEC).
    PRIME_CONSENT_SEAT_WAIT_SEC: float = Field(default=15.0, ge=0.0, le=30.0, alias="PRIME_CONSENT_SEAT_WAIT_SEC")
    PRIME_CONSENT_LLM_TIMEOUT_SEC: float = Field(default=20.0, gt=0.0, le=30.0, alias="PRIME_CONSENT_LLM_TIMEOUT_SEC")
    PRIME_CONSENT_MAX_TOKENS: int = Field(default=200, gt=0, le=600, alias="PRIME_CONSENT_MAX_TOKENS")
    # Self-monitor E1 act gate on Prime's own acts (orion/self_monitor/gate.py; app/prime_loop.py): an act
    # whose faculty (act.prime.<verb>, e.g. act.prime.predict) is held broken/unreliable in a fresh
    # orion:self_monitor:latest is not opened (track_record_failing), except one re-test probe per faculty
    # per SELF_MONITOR_PROBE_INTERVAL_SEC. Unknown/stale/absent state changes nothing. Shipped ON.
    SELF_MONITOR_ACT_GATE_ENABLED: bool = Field(default=True, alias="SELF_MONITOR_ACT_GATE_ENABLED")
    SELF_MONITOR_PROBE_INTERVAL_SEC: float = Field(default=21600.0, ge=0.0, alias="SELF_MONITOR_PROBE_INTERVAL_SEC")
    # Read-only grammar_acts read for the gate's probe and same-job bookkeeping. Empty: the in-process
    # memo alone decides (a restart may then probe once early).
    SELF_MONITOR_LEDGER_PG_DSN: str = Field(default="", alias="SELF_MONITOR_LEDGER_PG_DSN")

    # The curiosity replica (docs/superpowers/specs/2026-11-04-curiosity-replica-design.md; app/replica_loop.py):
    # instance #2 of the loop library. Answers Hub's plan requests on the replica arm: one model call per plan,
    # family curiosity.replica (tier 2, lane chat). Off: no listener, no bids; Hub's replica-arm runs then time
    # out to the uncertainty order. Shipped ON.
    CURIOSITY_REPLICA_ENABLED: bool = Field(default=True, alias="CURIOSITY_REPLICA_ENABLED")
    # The wait between ticks; a plan request wakes the loop at once.
    CURIOSITY_REPLICA_TICK_SEC: float = Field(default=15.0, gt=0.0, alias="CURIOSITY_REPLICA_TICK_SEC")
    # The no-model backoff's base (the library floors it at 60 s). A request's own deadline bounds retries.
    CURIOSITY_REPLICA_REFRACTORY_SEC: float = Field(default=0.0, ge=0.0, alias="CURIOSITY_REPLICA_REFRACTORY_SEC")
    # How often open plans are checked against their run's score (Postgres, read-only).
    CURIOSITY_REPLICA_SETTLE_INTERVAL_SEC: float = Field(default=300.0, gt=0.0, alias="CURIOSITY_REPLICA_SETTLE_INTERVAL_SEC")
    # The warm chat lane Prime uses (config/gpu_pool.yaml route `prime`); no new allocation.
    CURIOSITY_REPLICA_LLM_ROUTE: str = Field(default="prime", alias="CURIOSITY_REPLICA_LLM_ROUTE")
    # Most a try may wait for a seat; cut further so seat wait + generation end before Hub's deadline.
    CURIOSITY_REPLICA_SEAT_WAIT_SEC: float = Field(default=45.0, ge=0.0, le=300.0, alias="CURIOSITY_REPLICA_SEAT_WAIT_SEC")
    CURIOSITY_REPLICA_BID_LANE: str = Field(default="chat", alias="CURIOSITY_REPLICA_BID_LANE")
    CURIOSITY_REPLICA_MAX_TOKENS: int = Field(default=500, gt=0, alias="CURIOSITY_REPLICA_MAX_TOKENS")
    # Generation budget per plan (also the bid's cost ceiling). A request with less time left than this
    # (minus a 5 s answer margin) is answered "unfunded"/"model_failed" instead of tried.
    CURIOSITY_REPLICA_LLM_TIMEOUT_SEC: float = Field(default=40.0, gt=0.0, alias="CURIOSITY_REPLICA_LLM_TIMEOUT_SEC")
    # Hub (host network on athena): recent investigate runs for the frame (/curiosity/api/runs).
    CURIOSITY_REPLICA_HUB_URL: str = Field(default="http://100.92.216.81:8080", alias="CURIOSITY_REPLICA_HUB_URL")
    # Read-only Postgres: Hub's plan rows, the runs' spend-log scores, the newest chat-history digest.
    # Empty: those frame sections show as unavailable and open plans are never graded.
    CURIOSITY_REPLICA_PG_DSN: str = Field(default="", alias="CURIOSITY_REPLICA_PG_DSN")
    # Answers to Prime (design section 9, patch 4; orion/curiosity/replica/answers.py): support / challenge /
    # refuse a focus Prime queued or a regulate signal on curiosity, each an act. Off: stances are recorded on
    # the plan row only. Shipped ON.
    CURIOSITY_REPLICA_ANSWERS_ENABLED: bool = Field(default=True, alias="CURIOSITY_REPLICA_ANSWERS_ENABLED")
    # Holdback: refusals recorded but not applied; challenges recorded but not shown to Prime (deterministic draw).
    CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION: float = Field(default=0.2, ge=0.0, le=1.0,
                                                               alias="CURIOSITY_REPLICA_REFUSAL_HOLDBACK_FRACTION")
    CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION: float = Field(default=0.2, ge=0.0, le=1.0,
                                                                 alias="CURIOSITY_REPLICA_CHALLENGE_HOLDBACK_FRACTION")
    # Caps: answers in any 24 h, and the least time between two answers (the Prime-replica loop guard).
    CURIOSITY_REPLICA_ANSWERS_PER_DAY: int = Field(default=3, ge=0, alias="CURIOSITY_REPLICA_ANSWERS_PER_DAY")
    CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC: float = Field(default=1800.0, ge=0.0,
                                                           alias="CURIOSITY_REPLICA_ANSWER_REFRACTORY_SEC")

    # Optional: cross-service timeout hierarchy check (Orch HTTP); logging only, no dependency.
    ORION_MIND_TIMEOUT_SEC: float | None = Field(default=None, alias="ORION_MIND_TIMEOUT_SEC")

    @property
    def router_profiles_dir(self) -> Path:
        if (self.MIND_ROUTER_PROFILES_PATH or "").strip():
            return Path(self.MIND_ROUTER_PROFILES_PATH)
        return Path(__file__).resolve().parent / "config"


settings = Settings()

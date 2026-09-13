"""Production acknowledgement and replica/replay contracts against isolated PostgreSQL.

Set ORION_VISUAL_TEST_DATABASE_URL to a disposable database (never production).
The fixture creates an isolated schema; no existing tables are changed.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, text
from orion.schemas.reverie_visual import (
    ReverieVisualChainV1, ReverieVisualArtifactV1, VisualRunRequestV1,
    VisualBaselineEligibilityV1,
)

NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)


@pytest.fixture
def database(monkeypatch):
    from app import store
    url = os.environ.get("ORION_VISUAL_TEST_DATABASE_URL")
    if not url:
        pytest.skip("isolated PostgreSQL URL required")
    engine = create_engine(url)
    schema = "visual_test_" + uuid4().hex
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine.dispose()
    engine = create_engine(url, connect_args={"options": f"-c search_path={schema}"})
    root = Path(__file__).resolve().parents[3]
    with engine.begin() as conn:
        for filename in ("manual_migration_reverie_visual_chain.sql", "manual_migration_reverie_visual_attempt.sql"):
            conn.execute(text((root / "services/orion-sql-db" / filename).read_text()))
    monkeypatch.setattr(store, "_get_engine", lambda: engine)
    yield store, engine
    engine.dispose()


def production(store, tmp_path, chain_id="chain-1", at=NOW, content=b"real stored image bytes",
               context_selection=None, thermal_gate=None):
    from hashlib import sha256
    sha = sha256(content).hexdigest()
    path = tmp_path / f"{sha}.png"
    path.write_bytes(content)
    chain = ReverieVisualChainV1(chain_id=chain_id, created_at=at, context_selection=context_selection,
        chain_json={"artifact_sha256": sha, "production_receipt": None, "description": None,
                    **({"thermal_gate": thermal_gate} if thermal_gate else {})})
    artifact = ReverieVisualArtifactV1(chain_id=chain_id, sha256=sha, step_index=0,
        mime="image/png", bytes=len(content), path=str(path), created_at=at)
    assert store.persist_reverie_visual_chain(chain)
    return chain, artifact


def test_receipt_duplicate_bytes_advance_per_run_and_failures_do_not(database, tmp_path):
    store, engine = database
    assert store.load_visual_activity().history_status == "ok"
    assert store.load_visual_activity().last_success_at is None
    chain, artifact = production(store, tmp_path)
    assert store.load_visual_activity().last_success_at is None
    first = store.acknowledge_visual_production(chain, artifact)
    assert first is not None
    # No caption is required to count an actually persisted image.
    assert store.load_visual_activity().last_success_at == NOW
    second_chain, second_artifact = production(store, tmp_path, "chain-2", NOW + timedelta(hours=2))
    second = store.acknowledge_visual_production(second_chain, second_artifact)
    assert second.sha256 == first.sha256
    assert store.load_visual_activity().last_success_chain_id == "chain-2"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM reverie_visual_artifact")).scalar() == 1
    for reason in ("thermal_refused", "generation_failed", "run_deadline_exceeded"):
        assert store.persist_reverie_visual_chain(ReverieVisualChainV1(
            chain_id=reason, created_at=NOW + timedelta(hours=3), terminal_reason=reason,
            chain_json={"abandoned_chain_id": "chain-2"}))
    activity = store.load_visual_activity()
    assert activity.last_success_at == NOW + timedelta(hours=2)
    # New failed acknowledgements NEVER use the legacy join fallback.
    chain, artifact = production(store, tmp_path, "failed-ack", NOW + timedelta(hours=4), b"other bytes")
    assert store.persist_reverie_visual_artifact(artifact)
    Path(artifact.path).write_bytes(b"corrupted")
    assert store.acknowledge_visual_production(chain, artifact) is None
    assert store.load_visual_activity().last_success_chain_id == "chain-2"


def test_historical_predicate_requires_matching_sha_positive_bytes_and_terminal(database, tmp_path):
    store, engine = database
    chain, artifact = production(store, tmp_path)
    assert store.persist_reverie_visual_artifact(artifact)
    with engine.begin() as conn:
        conn.execute(text("UPDATE reverie_visual_chain SET chain_json=chain_json-'production_receipt'"))
    assert store.load_visual_activity().last_success_chain_id == "chain-1"
    with engine.begin() as conn:
        conn.execute(text("UPDATE reverie_visual_artifact SET bytes=0"))
    assert store.load_visual_activity().last_success_chain_id is None


def test_unavailable_is_not_empty_history(monkeypatch):
    from app import store
    def unavailable():
        raise RuntimeError("database offline")
    monkeypatch.setattr(store, "_get_engine", unavailable)
    assert store.load_visual_activity().history_status == "unavailable"


def test_claim_replay_concurrency_and_ambiguous_hold(database, tmp_path):
    store, _ = database
    need = VisualBaselineEligibilityV1(need_id="need-1", policy_id="visual_baseline.v1", observed_at=NOW, due_at=NOW)
    def claim(index):
        request = VisualRunRequestV1(dispatch_id=f"dispatch-{index}", visual_baseline=need)
        return store.claim_visual_attempt(request, retry_sec=600, now=NOW)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    winners = [attempt for attempt, replay in results if attempt]
    assert len(winners) == 1
    attempt = winners[0]
    assert any(replay and replay["outcome"] == "deferred_busy" for _, replay in results)
    assert claim(0)[0] is None  # identical dispatch does not regenerate
    store.finish_visual_attempt(attempt, {"outcome": "unknown", "ran": False})
    # Passing time is not proof a blocking GPU job stopped.
    assert store.claim_visual_attempt(VisualRunRequestV1(dispatch_id="later", visual_baseline=need),
        retry_sec=600, now=NOW + timedelta(days=2))[1]["reason"] == "attempt_unresolved"
    chain, artifact = production(store, tmp_path, attempt)
    assert store.acknowledge_visual_production(chain, artifact)
    # Positive receipt reconciles ambiguity; stale pending need is satisfied.
    assert store.claim_visual_attempt(VisualRunRequestV1(dispatch_id="reconcile", visual_baseline=need),
        retry_sec=600, now=NOW + timedelta(days=2))[1]["outcome"] == "already_satisfied"


def test_deferral_keeps_need_and_retry_gap(database):
    store, _ = database
    need = VisualBaselineEligibilityV1(need_id="need-1", policy_id="visual_baseline.v1", observed_at=NOW, due_at=NOW)
    request = VisualRunRequestV1(dispatch_id="d1", visual_baseline=need)
    attempt, _ = store.claim_visual_attempt(request, retry_sec=600, now=NOW)
    result = {"outcome": "deferred_thermal", "ran": True}
    store.finish_visual_attempt(attempt, result)
    assert store.claim_visual_attempt(request, retry_sec=600, now=NOW)[1] == result
    retry = VisualRunRequestV1(dispatch_id="d2", visual_baseline=need)
    assert store.claim_visual_attempt(retry, retry_sec=600, now=NOW)[1]["reason"] == "retry_cooldown"
    assert store.claim_visual_attempt(retry, retry_sec=600, now=NOW + timedelta(seconds=601))[0]


@pytest.mark.asyncio
async def test_http_thermal_retry_production_and_replay_keep_identity(database, tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from app import main
    from orion.reverie import baseline
    from orion.schemas.reverie_visual import VisualContextSelectionV1, VisualSourceV1
    store, _ = database
    vc = sys.modules['app.visual_chain']
    clock = [NOW]
    class Clock:
        @staticmethod
        def now(tz=None):
            return clock[0]
    monkeypatch.setattr(main, 'datetime', Clock)
    policy = baseline.VisualBaselinePolicy(enabled=True)
    monkeypatch.setattr(baseline, 'load_baseline_policy', lambda: policy)
    # Validator has its own clock; validate frozen request time explicitly.
    original_validate = baseline.validate_eligibility
    monkeypatch.setattr(baseline, 'validate_eligibility', lambda value, **kw: original_validate(value, now=clock[0], **kw))
    monkeypatch.setattr(vc.settings, 'visual_chain_enabled', False)
    monkeypatch.setattr(vc.settings, 'thermal_gate_enabled', True)
    hot = [True]
    async def thermal():
        return SimpleNamespace(state='hot' if hot[0] else 'cool', temp_c=30 if hot[0] else 23,
            age_sec=1, reason='test_reading', allows_gpu_work=not hot[0], degraded=False)
    monkeypatch.setattr(vc, 'evaluate_thermal_gate', thermal)
    class Bus:
        def __init__(self, **kw): pass
        async def connect(self): pass
        async def close(self): pass
    monkeypatch.setattr('orion.core.bus.async_service.OrionBusAsync', Bus)
    calls = []
    async def body(bus, *, attempt_id, thermal_gate, run_request, **kw):
        calls.append(attempt_id)
        chain, artifact = production(store, tmp_path, attempt_id, clock[0])
        chain.context_selection = VisualContextSelectionV1(source_kind='memory',
            source=VisualSourceV1(source_id='memory-real-id', text='private source text'))
        chain.chain_json['thermal_gate'] = thermal_gate
        receipt = store.acknowledge_visual_production(chain, artifact)
        chain.chain_json['production_receipt'] = receipt.model_dump(mode='json')
        return chain
    monkeypatch.setattr(vc, '_run_visual_chain_body', body)
    need = VisualBaselineEligibilityV1(need_id='one-need', policy_id=policy.policy_id, observed_at=NOW, due_at=NOW)
    request = VisualRunRequestV1(dispatch_id='first', proposal_id='proposal', decision_id='decision',
        correlation_id='correlation', visual_baseline=need)
    refused = json.loads((await main.visual_chain_run_once(request)).body)
    assert refused['outcome'] == 'deferred_thermal'
    assert refused['execution_receipt']['source_selection_status'] == 'source_selection_not_reached'
    assert not calls
    clock[0] += timedelta(seconds=601)
    hot[0] = False
    request = request.model_copy(update={'dispatch_id': 'second', 'visual_baseline': need.model_copy(update={'observed_at': clock[0]})})
    result = json.loads((await main.visual_chain_run_once(request)).body)
    assert result['outcome'] == 'produced'
    assert result['artifact_persisted']
    receipt = result['execution_receipt']
    assert receipt['request']['visual_baseline']['need_id'] == 'one-need'
    assert receipt['request']['correlation_id'] == 'correlation'
    assert receipt['source_refs'] == ['memory-real-id']
    assert 'private source text' not in json.dumps(result)
    clock[0] += timedelta(days=2)  # Authorization is stale; completed replay is not new work.
    replay = json.loads((await main.visual_chain_run_once(request)).body)
    assert replay == result
    changed = request.model_copy(update={"proposal_id": "different-proposal"})
    mismatch = json.loads((await main.visual_chain_run_once(changed)).body)
    assert mismatch["reason"] == "dispatch_request_mismatch"
    assert len(calls) == 1
    assert store.load_visual_activity().last_success_chain_id == result['chain_id']


def test_settled_source_identity_is_preserved_and_full_thought_revalidated(database):
    store, engine = database
    from orion.schemas.reverie import SpontaneousThoughtV1
    from orion.schemas.thought import CoalitionSnapshotV1
    now = datetime.now(timezone.utc)
    coalition = CoalitionSnapshotV1(attended_node_ids=['real-node'], open_loop_ids=['real-loop'],
        selected_open_loop_id='real-loop', generated_at=now)
    thought = SpontaneousThoughtV1(thought_id='real-thought', correlation_id='real-correlation',
        created_at=now, coalition=coalition, interpretation='A grounded thought about the real mesh and its changing connections.',
        evidence_refs=['real-loop'])
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE substrate_reverie_thought (thought_id text, created_at timestamptz, interpretation text, thought_json jsonb)'))
        conn.execute(text('CREATE TABLE substrate_reverie_chain (chain_id text, created_at timestamptz, chain_json jsonb)'))
        conn.execute(text('INSERT INTO substrate_reverie_thought VALUES (:id,:at,:interpretation,CAST(:payload AS jsonb))'),
            {'id':thought.thought_id,'at':now,'interpretation':thought.interpretation,'payload':thought.model_dump_json()})
    assert store.load_latest_reverie_interpretation(max_age_sec=900) is None
    with engine.begin() as conn:
        conn.execute(text('INSERT INTO substrate_reverie_chain VALUES (:id,:at,CAST(:payload AS jsonb))'),
            {'id':'verified-settled-chain','at':now,'payload':'{"thought_ids":["real-thought"]}'})
    context = store.load_latest_reverie_interpretation(max_age_sec=900, char_limit=18)
    assert context.thought_id == 'real-thought'
    assert context.text_chain_id == 'verified-settled-chain'
    assert context.thought_correlation_id == 'real-correlation'
    assert context.coalition == coalition
    assert len(context.text) <= 18
    # Original valid thought is checked before shortening; short context remains valid.
    with engine.begin() as conn:
        conn.execute(text("UPDATE substrate_reverie_thought SET created_at=now()-interval '2 days'"))
    assert store.load_latest_reverie_interpretation(max_age_sec=900) is None


@pytest.mark.asyncio
async def test_only_selected_source_is_credited_and_store_restores_selection(database, tmp_path, monkeypatch):
    from app import visual_chain as vc
    from orion.schemas.reverie_visual import ReverieVisualContextV1, VisualSourceV1
    from orion.schemas.thought import CoalitionSnapshotV1
    store, engine = database
    context = ReverieVisualContextV1(text='A real thought about the mesh.', thought_id='thought-id',
        thought_correlation_id='thought-correlation', thought_created_at=NOW, text_chain_id='settled-id',
        coalition=CoalitionSnapshotV1(attended_node_ids=['node'], open_loop_ids=['loop'],
            selected_open_loop_id='loop', generated_at=NOW), evidence_refs=['loop'])
    rotation = [0]
    monkeypatch.setattr(vc, 'load_latest_visual_chain_continuity_state', lambda **kw: ('previous scene', 0, rotation[0], 'prior-chain-id'))
    monkeypatch.setattr(vc, 'load_latest_reverie_interpretation', lambda **kw: context)
    monkeypatch.setattr(vc, 'load_latest_self_study_reflection', lambda **kw: VisualSourceV1(source_id='entry-id', text='Observed traffic changed.'))
    monkeypatch.setattr(vc, 'load_latest_memory_crystallization', lambda **kw: VisualSourceV1(source_id='crystal-id', text='A reviewed memory.'))
    monkeypatch.setattr(vc.settings, 'visual_chain_interpretation_enabled', False)
    monkeypatch.setattr(vc.settings, 'thermal_gate_enabled', False)
    monkeypatch.setattr(vc.settings, 'visual_chain_storage_dir', str(tmp_path))
    import struct
    png = b'\x89PNG\r\n\x1a\n' + struct.pack('>I',13) + b'IHDR' + struct.pack('>II',32,32)
    monkeypatch.setattr(vc, 'call_diffusion_generate', lambda *a, **kw: png)
    def no_caption(*a, **kw): raise RuntimeError('caption unavailable')
    monkeypatch.setattr(vc, 'upload_to_percept_store', no_caption)
    for index, kind, source_id in [(0,'reverie','thought-id'),(1,'self_study','entry-id'),(2,'memory','crystal-id')]:
        rotation[0] = index
        chain = await vc.run_visual_chain_once(bus=None)
        assert chain.chain_json['production_receipt']
        selected = chain.context_selection
        assert selected.source_kind == kind
        assert selected.continuity.source_id == 'prior-chain-id'
        assert (selected.reverie.thought_id if selected.reverie else selected.source.source_id) == source_id
        if kind != 'reverie':
            assert selected.reverie is None  # loaded-but-unused thought is not credited
        with engine.connect() as conn:
            row = conn.execute(text('SELECT chain_json FROM reverie_visual_chain WHERE chain_id=:id'), {'id':chain.chain_id}).scalar()
        assert ReverieVisualChainV1(chain_id=chain.chain_id, chain_json=row).context_selection == selected


def test_positive_receipt_clears_activity_hold_and_same_dispatch_reconciles(database, tmp_path):
    store, engine = database
    request = VisualRunRequestV1(dispatch_id="crash-after-production", proposal_id="original")
    attempt, _ = store.claim_visual_attempt(request, retry_sec=600, now=NOW)
    assert store.load_visual_activity().active_attempt_id == attempt
    store.finish_visual_attempt(attempt, {"outcome": "unknown", "ran": False})
    assert store.load_visual_activity().active_attempt_id == attempt
    from orion.schemas.reverie_visual import VisualContextSelectionV1, VisualSourceV1
    selection = VisualContextSelectionV1(source_kind="memory",
        source=VisualSourceV1(source_id="memory-recovered", text="private source content"))
    chain, artifact = production(store, tmp_path, attempt, context_selection=selection,
                                thermal_gate={"reason": "test_cool_receipt"})
    assert store.acknowledge_visual_production(chain, artifact)
    # The read projection must unblock scheduling without needing another claim.
    assert store.load_visual_activity().active_attempt_id is None
    # It stays read-only; replay performs the bounded durable reconciliation.
    with engine.connect() as conn:
        assert conn.execute(text("SELECT outcome FROM reverie_visual_attempt")).scalar() == "unknown"
    replay = store.replay_visual_attempt(request)
    assert replay["outcome"] == "produced"
    assert replay["attempt_id"] == attempt
    assert replay["execution_receipt"]["request"] == request.model_dump(mode="json")
    assert replay["execution_receipt"]["source_refs"] == ["memory-recovered"]
    assert replay["execution_receipt"]["thermal_gate"] == {"reason": "test_cool_receipt"}
    assert "private source content" not in str(replay)
    assert store.replay_visual_attempt(request) == replay
    assert store.claim_visual_attempt(request, retry_sec=600, now=NOW + timedelta(days=2)) == (None, replay)
    changed = request.model_copy(update={"proposal_id": "other"})
    assert store.replay_visual_attempt(changed)["reason"] == "dispatch_request_mismatch"
    assert store.replay_visual_attempt(request) == replay


def test_mismatched_positive_receipt_does_not_release_unknown_attempt(database, tmp_path):
    store, engine = database
    request = VisualRunRequestV1(dispatch_id="unresolved")
    attempt, _ = store.claim_visual_attempt(request, retry_sec=600, now=NOW)
    chain, artifact = production(store, tmp_path, attempt)
    assert store.acknowledge_visual_production(chain, artifact)
    with engine.begin() as conn:
        conn.execute(text("UPDATE reverie_visual_chain SET chain_json=jsonb_set(chain_json, '{production_receipt,attempt_id}', to_jsonb(CAST(:value AS text)))"), {"value": "different"})
    assert store.load_visual_activity().active_attempt_id == attempt
    assert store.replay_visual_attempt(request)["outcome"] == "unknown"
    later = VisualRunRequestV1(dispatch_id="days-later")
    assert store.claim_visual_attempt(later, retry_sec=600, now=NOW + timedelta(days=2))[1]["reason"] == "attempt_unresolved"


def test_ack_retry_rechecks_canonical_file_and_never_overwrites_receipt(database, tmp_path):
    store, engine = database
    chain, artifact = production(store, tmp_path)
    receipt = store.acknowledge_visual_production(chain, artifact)
    alternate = tmp_path / "alternate.png"
    alternate.write_bytes(Path(artifact.path).read_bytes())
    alternate_artifact = artifact.model_copy(update={"path": str(alternate), "created_at": NOW + timedelta(days=1)})
    assert store.acknowledge_visual_production(chain, alternate_artifact) == receipt
    Path(receipt.path).write_bytes(b"corrupt canonical bytes")
    assert store.acknowledge_visual_production(chain, alternate_artifact) is None
    with engine.connect() as conn:
        stored = conn.execute(text("SELECT chain_json->'production_receipt' FROM reverie_visual_chain WHERE chain_id=:id"), {"id": chain.chain_id}).scalar()
    assert stored == receipt.model_dump(mode="json")


def test_optional_replay_before_attempt_table_migration(database):
    store, engine = database
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE reverie_visual_attempt RENAME TO visual_attempt_not_installed"))
    assert store.replay_visual_attempt(VisualRunRequestV1(dispatch_id="ordinary-extra")) is None
    # New baseline claims still require their durable claim table.
    with pytest.raises(Exception):
        store.claim_visual_attempt(VisualRunRequestV1(dispatch_id="new-claim"), retry_sec=600, now=NOW)

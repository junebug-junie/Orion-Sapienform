from __future__ import annotations

import json
import subprocess
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
MIND_PROVENANCE_JS = HUB_ROOT / "static" / "js" / "mind_provenance.js"


def _run_node(script: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=HUB_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout.strip())


def test_fail_open_fixture_normalizer() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = OrionMindProvenance.MIND_PROVENANCE_FIXTURES.failOpen;
const prov = OrionMindProvenance.normalizeMindRunProvenance(run);
const rows = OrionMindProvenance.normalizeMindPhaseRows(run);
const callouts = OrionMindProvenance.normalizeMindDerailments(run);
const html = OrionMindProvenance.renderMindDerailmentCallouts(callouts);
console.log(JSON.stringify({{
  cognition_path: prov.cognition_path,
  llm_attempted: prov.llm_attempted,
  failed_phase: prov.failed_phase,
  semantic_status: rows.find(r => r.phase === 'semantic_synthesis')?.status,
  has_fallback_row: rows.some(r => r.phase === 'deterministic_fallback'),
  callout_count: callouts.length,
  has_off_rails: html.includes('Where it went off rails'),
}}));
"""
    )
    assert out["cognition_path"] == "llm_fail_open_to_deterministic"
    assert out["llm_attempted"] is True
    assert out["failed_phase"] == "semantic_synthesis"
    assert out["semantic_status"] == "failed"
    assert out["has_fallback_row"] is True
    assert out["callout_count"] >= 1
    assert out["has_off_rails"] is True


def test_orch_http_failed_fixture_normalizer() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = OrionMindProvenance.MIND_PROVENANCE_FIXTURES.orchHttpFailed;
const prov = OrionMindProvenance.normalizeMindRunProvenance(run);
const rows = OrionMindProvenance.normalizeMindPhaseRows(run);
const callouts = OrionMindProvenance.normalizeMindDerailments(run);
const html = OrionMindProvenance.renderMindDerailmentCallouts(callouts);
console.log(JSON.stringify({{
  cognition_path: prov.cognition_path,
  orch_http_failed: prov.orch_http_failed,
  has_orch_row: rows.some(r => r.phase === 'orch_mind_http'),
  callout_id: callouts[0] && callouts[0].id,
  title: callouts[0] && callouts[0].title,
  has_off_rails: html.includes('Where it went off rails'),
}}));
"""
    )
    assert out["cognition_path"] == "orch_mind_http_failed"
    assert out["orch_http_failed"] is True
    assert out["has_orch_row"] is True
    assert out["callout_id"] == "orch_mind_http_failed"
    assert "Orch timed out" in (out["title"] or "")
    assert out["has_off_rails"] is True


def test_success_fixture_normalizer() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = OrionMindProvenance.MIND_PROVENANCE_FIXTURES.success;
const prov = OrionMindProvenance.normalizeMindRunProvenance(run);
const rows = OrionMindProvenance.normalizeMindPhaseRows(run);
const callouts = OrionMindProvenance.normalizeMindDerailments(run);
const html = OrionMindProvenance.renderMindDerailmentCallouts(callouts);
console.log(JSON.stringify({{
  cognition_path: prov.cognition_path,
  authorized: prov.authorized_for_stance_use,
  error_callouts: callouts.filter(c => c.severity === 'error').length,
  semantic_ok: rows.find(r => r.phase === 'semantic_synthesis')?.status,
  appraisal_ok: rows.find(r => r.phase === 'active_frontier_judge')?.status,
  stance_ok: rows.find(r => r.phase === 'stance_handoff')?.status,
  no_derailment_card: html.includes('without derailment'),
}}));
"""
    )
    assert out["cognition_path"] == "llm_synthesis"
    assert out["authorized"] is True
    assert out["error_callouts"] == 0
    assert out["semantic_ok"] == "ok"
    assert out["appraisal_ok"] == "ok"
    assert out["stance_ok"] == "ok"
    assert out["no_derailment_card"] is True


def test_success_fixture_renders_brief_item_cards() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = OrionMindProvenance.MIND_PROVENANCE_FIXTURES.success;
const html = OrionMindProvenance.renderMindProvenanceSections(run);
console.log(JSON.stringify({{
  has_items_section: html.includes('Mind synthesis items'),
  has_semantic_card: html.includes('shared evening moment'),
  has_frontier_card: html.includes('relationship_opportunity'),
  has_provenance: html.includes('Cognition path'),
}}));
"""
    )
    assert out["has_items_section"] is True
    assert out["has_semantic_card"] is True
    assert out["has_frontier_card"] is True
    assert out["has_provenance"] is True


def test_shadow_fixture_renders_shadow_fields() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = OrionMindProvenance.MIND_PROVENANCE_FIXTURES.shadowSynthesis;
const prov = OrionMindProvenance.normalizeMindRunProvenance(run);
const html = OrionMindProvenance.renderMindBriefItems(run);
console.log(JSON.stringify({{
  cognition_path: prov.cognition_path,
  has_shadow_card: html.includes('Shadow synthesis'),
  has_attention: html.includes('evening plans with Amanda'),
  has_projection_refs: html.includes('projection:relationship:0'),
}}));
"""
    )
    assert out["cognition_path"] == "deterministic_shadow"
    assert out["has_shadow_card"] is True
    assert out["has_attention"] is True
    assert out["has_projection_refs"] is True


def test_fail_open_fixture_phase_rows_expose_semantic_filter_counts() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = {{
  result_jsonb: {{
    brief: {{
      machine_contract: {{
        "mind.llm_fail_open_to_deterministic": true,
        "mind.semantic_claim_count": 0,
        "mind.phase_telemetry": [{{
          phase_name: "semantic_synthesis",
          route: "quick",
          ok: true,
          parse_ok: true,
          validation_ok: false,
          status: "filtered",
          error: "semantic_synthesis_empty",
          raw_claim_count: 2,
          retained_claim_count: 0,
          filtered_claim_count: 2,
          filter_reasons_by_count: {{ unsupported_or_weak: 2 }},
          authorization_reason: "semantic_synthesis_empty",
        }}],
      }},
    }},
  }},
}};
const rows = OrionMindProvenance.normalizeMindPhaseRows(run);
const sem = rows.find(r => r.phase === 'semantic_synthesis');
console.log(JSON.stringify({{
  retained: sem && sem.retained_claim_count,
  raw: sem && sem.raw_claim_count,
  error_has_counts: sem && String(sem.error || '').includes('raw=2'),
}}));
"""
    )
    assert out["retained"] == 0
    assert out["raw"] == 2
    assert out["error_has_counts"] is True


def test_fail_open_fixture_still_renders_provenance_sections() -> None:
    out = _run_node(
        f"""
const fs = require('fs');
const src = fs.readFileSync({json.dumps(str(MIND_PROVENANCE_JS))}, 'utf8');
eval(src);
const run = OrionMindProvenance.MIND_PROVENANCE_FIXTURES.failOpen;
const html = OrionMindProvenance.renderMindProvenanceSections(run);
console.log(JSON.stringify({{
  has_off_rails: html.includes('Where it went off rails'),
  has_phase_table: html.includes('Phase provenance'),
  has_fallback_row: html.includes('deterministic_fallback'),
  has_orch_http: html.includes('orch_mind_http'),
}}));
"""
    )
    assert out["has_off_rails"] is True
    assert out["has_phase_table"] is True
    assert out["has_fallback_row"] is True
    assert out["has_orch_http"] is False

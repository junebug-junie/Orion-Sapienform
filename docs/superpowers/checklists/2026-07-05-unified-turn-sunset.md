# Unified-turn Brain shim sunset checklist

**Do not remove Brain `chat_general` mode until every item below is green.**

Reference: [unified-turn design spec §13](../specs/2026-07-05-orion-unified-turn-design.md#13-brain-shim-sunset)

Last updated: 2026-07-05

---

## Sunset criteria

- [ ] **Mesh-honesty eval suite** — no location/belief claims in `final_text` without matching `grammar_event_id` in outcome molecule
- [ ] **Repair/refusal parity eval suite** — unified path matches Brain repair and refusal behavior on shared fixtures
- [ ] **Relational stance parity** — `test_stance_react_relational_survives_compressor` green on companion-turn fixtures
- [ ] **Pollution firewall** — `services/orion-hub/tests/test_unified_orion_turn_pollution_firewall.py` green
- [ ] **Layer attribution evals** — `orion/harness/evals/test_layer_attribution.py` green (5a→5b, 5b→5c, N→N+1 strain)
- [ ] **agent-claude tool parity** — unified path completes same repo-edit eval fixtures with grammar trace
- [ ] **Operator soak** — `ORION_UNIFIED_TURN_ENABLED=true` for 14 consecutive days without rollback
- [ ] **Cost ceiling** — median brain tokens/turn ≤ 1.5× Brain mode on eval corpus

---

## Verification commands

```bash
pytest orion/thought/tests/test_stance_react_relational_survives_compressor.py -q
pytest services/orion-hub/tests/test_unified_orion_turn_pollution_firewall.py -q
pytest orion/harness/evals/test_layer_attribution.py -q
pytest orion/thought/tests/test_trust_rupture_threshold_frozen.py -q
```

---

## Rollback

If any criterion fails during soak:

1. Set `ORION_UNIFIED_TURN_ENABLED=false` in Hub env
2. Restart `orion-hub` and `orion-harness-governor`
3. File incident with failing criterion and correlation IDs

---

## Notes

- Brain mode remains the `chat_general` speech shim until this checklist is fully green.
- Changing `TRUST_RUPTURE_DEFER_THRESHOLD` or `FINALIZE_QUICK_GATE_EPSILON` requires re-running their eval corpora and updating frozen gate tests.

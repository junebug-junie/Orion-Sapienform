"""Faculties Orion has but cannot grade from their own bets yet, with the fix that would unlock each.

From the 2026-10-20 live audit (design spec, "Where Orion's bets live today"). A faculty here
appears in the state as ``ungradeable`` with its reason instead of silently missing. When a source
fix lands and an adapter grades it, its adapter's records replace the entry (``reduce_bets`` skips a
catalog entry whose faculty already has bets).
"""

from __future__ import annotations

from orion.self_monitor.record import Ungradeable

UNGRADEABLE: tuple[Ungradeable, ...] = (
    Ungradeable(
        "reverie.expectation", "substrate_reverie_thought", "ungradeable:never_settled",
        "984 of 988 expectations went unscored in 7 days (10-20) and none states a confidence; "
        "the claims are near-tautological ('the laptop will still be on the desk')",
    ),
    Ungradeable(
        "dream.hypothesis", "dream_hypothesis", "ungradeable:self_graded",
        "graded only by whether Orion later adopts it (dream vs random-control arm); no confidence stored",
    ),
    Ungradeable(
        "curiosity.hypothesis", "curiosity_run_outcomes", "ungradeable:self_graded",
        "priors are re-judged by Orion after investigating; no outside settlement",
    ),
    Ungradeable(
        # Corrected 2026-11-03: the 10-20 audit read these as attention asks with a salience bug. All
        # 63 `orion_is_asking` outcomes are memory-confirmation cards (orion/memory/episode/
        # confirmation.py); no salience or confidence exists before the ask, so there is nothing to
        # copy -- episode_memory.strength is a per-purpose constant the answer itself changes.
        "memory.confirmation", "attention_loop_outcome", "ungradeable:no_stated_confidence",
        "Orion makes no bet before asking Juniper to confirm a memory; add a confirm-chance bet "
        "(orion-prediction) when the card opens",
    ),
    Ungradeable(
        "vision.same_person", "vision_individual_pair_verdict", "ungradeable:source_bug",
        "verdict copies the ask's face_cosine in only 10/16 rows; copy it from orion_ask.evidence_refs",
    ),
    Ungradeable(
        "body.gpu_power", "power_intent_settled", "ungradeable:source_bug",
        "expected_watts filled on 7/41 settled intents (10-20); record it at intent time",
    ),
    Ungradeable(
        "chat.reply_quality", "chat_response_feedback", "ungradeable:no_stated_confidence",
        "8 thumbs ever and no prediction; substrate_action_ratings.predicted_rating has 0 rows",
    ),
    Ungradeable(
        "energy.bill", "energy_reconcile", "ungradeable:no_stated_confidence",
        "Orion makes no forward bill bet; add a bill series to orion-prediction",
    ),
)

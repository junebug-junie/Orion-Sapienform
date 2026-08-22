(function (global) {
  // Pure view-model logic for the Pending Attention / "Cognitive Loops" card
  // (app.js::renderCognitiveLoopCard). Kept as a standalone module (mirrors
  // container-bringup-ui.js/thought-process.js) so the chronic_pressure vs
  // resolvable branching -- which buttons show, which badge/note text renders --
  // is unit-testable without a DOM harness. app.js wires this to real elements.
  //
  // card_kind == 'chronic_pressure' means a reverie/substrate-broadcast loop
  // re-selected every tick by design (see orion/schemas/attention_salience.py's
  // PendingCardKindV1 docstring) -- Resolve/Dismiss on one would falsely mark
  // still-live system pressure as closed, so those actions must not render.
  function cognitiveLoopCardViewModel(card) {
    const isChronic = !!card && card.card_kind === "chronic_pressure";
    const recurrenceCount = (card && card.recurrence_count) || 0;
    return {
      isChronic: isChronic,
      borderClass: isChronic ? "border-amber-800" : "border-purple-800",
      showChronicBadge: isChronic,
      showActions: !isChronic,
      chronicNoteText: isChronic
        ? "Recurring " + recurrenceCount + "x -- ongoing system state, not a pending decision."
        : null,
    };
  }

  const api = { cognitiveLoopCardViewModel: cognitiveLoopCardViewModel };

  global.OrionCognitiveLoopCard = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);

(function (global) {
  // Pure label logic for the Container bring-up result <details>/<summary> toggle. Kept as a
  // standalone module (mirrors workflow-ui.js) so the open/closed label text is unit-testable
  // without a DOM harness -- app.js wires it to the real <details> element's 'toggle' event.
  function summaryLabelForOpenState(isOpen) {
    return isOpen ? 'Result (click to collapse)' : 'Result (click to expand)';
  }

  const api = { summaryLabelForOpenState };

  global.OrionContainerBringupUI = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

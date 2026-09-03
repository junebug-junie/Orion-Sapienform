(function (global) {
  // Pure elapsed-time formatting for the chat turn timer (app.js::paintTurnTimer).
  // Kept as a standalone module (mirrors container-bringup-ui.js/cognitive-loop-card.js)
  // so the sub-minute vs minute-rollover branching is unit-testable without a DOM
  // harness. app.js owns the interval and the element; this owns only the string.
  function formatTurnElapsed(elapsedMs) {
    const ms = Number(elapsedMs);
    // A clock that jumped backwards (NTP step, sleep/wake) must not render NaN
    // or a negative duration into the status bar.
    const secs = Number.isFinite(ms) && ms > 0 ? ms / 1000 : 0;
    if (secs < 60) return `${secs.toFixed(1)}s`;
    const mins = Math.floor(secs / 60);
    const rest = Math.floor(secs - mins * 60);
    return `${mins}m ${String(rest).padStart(2, '0')}s`;
  }

  const api = { formatTurnElapsed };

  global.OrionTurnTimer = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

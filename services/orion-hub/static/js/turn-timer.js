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
    // Branch on the value we are about to DISPLAY, not the raw one. Testing
    // `secs < 60` and then rendering `toFixed(1)` lets 59.95s..59.99s pass the
    // "under a minute" check and print "60.0s" -- the exact string the rollover
    // exists to prevent.
    const shown = Math.round(secs * 10) / 10;
    if (shown < 60) return `${shown.toFixed(1)}s`;
    const whole = Math.floor(shown);
    const mins = Math.floor(whole / 60);
    return `${mins}m ${String(whole - mins * 60).padStart(2, '0')}s`;
  }

  const api = { formatTurnElapsed };

  global.OrionTurnTimer = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

(function (global) {
  // Pure logic for the Vision panel's "Carbon (affect snapshot)" dropdown
  // view -- deciding what to render from a GET /api/vision/affect-ambient/
  // status response. Kept as a standalone module (mirrors
  // container-bringup-ui.js/workflow-ui.js) so this is unit-testable
  // without a DOM harness; app.js wires it to the real container element.
  //
  // Bug fix, 2026-08-22: this view used to render once on dropdown
  // selection and never again -- a real ambient/manual capture landing
  // seconds later was invisible until the dropdown was re-selected.
  //
  // Design note (review finding, round 2): an earlier draft of this fix
  // added a change-detection dedup key to skip "unchanged" repaints. That
  // key kept reproducing variations of the exact bug it was meant to fix
  // -- it couldn't tell a stale "in progress" snapshot from a just-
  // completed one, treated a sustained fetch failure as identical to the
  // very first "unavailable" render (so recovery never repainted), and
  // let the displayed elapsed-time text ("2m ago") go stale indefinitely
  // once nothing else about the status object changed. This version has
  // NO dedup: app.js (see repaintCarbonAffectSnapshot) just repaints
  // unconditionally on every poll while this view is active. A plain
  // innerHTML swap of a small text panel every ~15s has no visible cost
  // -- text replaces text in one paint, no interim blank state -- so
  // there was never a real reason to avoid it here the way
  // carbonLiveLastSha256 avoids re-fetching a full JPEG for the sibling
  // "Carbon (live)" view (that one's dedup is about bandwidth, not
  // flicker, and stays justified).

  function escapeHtml(value) {
    // Deliberately self-contained rather than reusing app.js's own
    // escapeHtml() (app.js:1632) -- this module runs standalone under
    // `node --test`, same as container-bringup-ui.js, with no dependency
    // on app.js's internals. Same three replacements as that helper
    // (review finding, round 2: an earlier draft here only escaped "<",
    // weaker than the helper already used elsewhere in app.js for the
    // same kind of user-facing text) -- also now applied to last_error,
    // which the pre-existing code never escaped at all.
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function formatAgoSeconds(deltaSec) {
    const clamped = Math.max(0, deltaSec);
    if (clamped < 60) return `${Math.round(clamped)}s ago`;
    if (clamped < 3600) return `${Math.round(clamped / 60)}m ago`;
    return `${Math.round(clamped / 3600)}h ago`;
  }

  function renderAffectSnapshotHtml(status, nowEpochSeconds) {
    if (!status) {
      return '<div class="text-gray-500 text-xs text-center px-4">Affect status unavailable.</div>';
    }
    if (!status.last_attempt_at) {
      return '<div class="text-gray-500 text-xs text-center px-4">No affect check has run yet -- use "Check now" or the ambient toggle.</div>';
    }
    const kindLabel = status.last_trigger === "manual" ? "manual check" : "ambient tick";
    if (status.tick_in_progress) {
      // Review finding, round 1: this view used to show nothing during
      // an in-flight capture (which can take up to ~195s per
      // vision_affect_ambient.py's own docstring) -- just stale leftover
      // content with no signal a new attempt was even running.
      return `<div class="text-gray-400 text-xs text-center px-4">Capturing now (${kindLabel})… this can take up to ~195s.</div>`;
    }
    const when = formatAgoSeconds(nowEpochSeconds - status.last_attempt_at);
    if (status.last_result_ok && status.last_raw_response) {
      return `
            <div class="w-full h-full overflow-y-auto p-4 text-sm text-gray-200 whitespace-pre-wrap">
              <div class="text-[11px] text-gray-500 mb-2">Last ${kindLabel}, ${when}</div>
              ${escapeHtml(status.last_raw_response)}
            </div>`;
    }
    if (status.last_result_ok) {
      // Review finding, round 1: last_result_ok=true with an empty
      // raw_response used to fall into the SAME "had no successful
      // result" message as a real failure, contradicting the backend's
      // own ok=true and giving no way to tell the two apart.
      return `<div class="text-gray-500 text-xs text-center px-4">Last ${kindLabel} (${when}) succeeded but returned no text.</div>`;
    }
    const errorSuffix = status.last_error ? ": " + escapeHtml(status.last_error) : "";
    return `<div class="text-gray-500 text-xs text-center px-4">Last ${kindLabel} (${when}) had no successful result${errorSuffix}.</div>`;
  }

  function createLatestWinsGate() {
    // Tracks "only the most recently issued token wins" -- app.js uses
    // this to drop a stale, later-resolving fetch response when a newer
    // fetch was issued after it but happens to resolve first (review
    // finding, round 3: the immediate fetch on dropdown selection and the
    // pre-existing 15s interval's own fetch can share the same view-
    // generation and race each other; the view-generation guard alone
    // only protects against a VIEW SWITCH, not against two same-view
    // fetches resolving out of order).
    let latest = 0;
    return {
      issue() {
        latest += 1;
        return latest;
      },
      isCurrent(token) {
        return token === latest;
      },
    };
  }

  const api = { renderAffectSnapshotHtml, formatAgoSeconds, escapeHtml, createLatestWinsGate };

  global.OrionCarbonAffectSnapshot = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);

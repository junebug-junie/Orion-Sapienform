(function (global) {
  // Pure logic for the Vision panel's polled "live" camera views
  // ("Carbon (live)", "Walkway (live)" -- 2026-10-15). Kept standalone, like
  // carbon-affect-snapshot.js, so it runs under `node --test` without a DOM;
  // app.js (renderLiveFrame) wires it to the real container.
  //
  // The point of this module: a frozen picture must never pass as live.
  // The backend (GET /api/vision/frames/latest) reports age_sec/stale from
  // Hub's own clock; this turns that into a visible LIVE / STALE badge and
  // dims the image once the stream stops moving.

  // Dropdown value -> stream + poll cadence. Walkway publishes ~2 frames/s;
  // carbon only every ~5s, so polling it faster would just repeat frames.
  const LIVE_FRAME_VIEWS = {
    "carbon-live": {
      streamId: "carbon",
      label: "Carbon",
      pollMs: 5000,
      hint: "confirm orion-vision-retina is running on carbon.",
    },
    "walkway-live": {
      streamId: "walkway",
      label: "Walkway",
      pollMs: 1500,
      hint: "confirm orion-vision-retina-walkway is running on athena.",
    },
  };

  function formatAge(ageSec) {
    if (typeof ageSec !== "number" || !Number.isFinite(ageSec)) return "unknown";
    if (ageSec < 1) return "just now";
    if (ageSec < 90) return `${Math.round(ageSec)}s ago`;
    if (ageSec < 5400) return `${Math.round(ageSec / 60)}m ago`;
    return `${Math.round(ageSec / 3600)}h ago`;
  }

  // meta: parsed JSON from /api/vision/frames/latest, or null on a failed
  // fetch. Returns what the panel should show.
  function liveFrameState(meta, view) {
    const label = (view && view.label) || "camera";
    const hint = (view && view.hint) || "";
    if (!meta || !meta.available) {
      return {
        kind: "none",
        message: `No frame from ${label} yet${hint ? " -- " + hint : "."}`,
        badgeText: "",
      };
    }
    const age = formatAge(meta.age_sec);
    if (meta.stale) {
      return {
        kind: "stale",
        message: "",
        badgeText: `STALE · last frame ${age}`,
        badgeClass: "bg-red-600/90 text-white",
        imageClass: "opacity-40 grayscale",
      };
    }
    return {
      kind: "live",
      message: "",
      badgeText: `LIVE · ${age}`,
      badgeClass: "bg-emerald-600/90 text-white",
      imageClass: "",
    };
  }

  // Only pull new image bytes when the backend has a different frame than
  // the one on screen -- the badge still refreshes every poll regardless.
  function shouldFetchImage(meta, shownFrameId) {
    return Boolean(meta && meta.available && meta.frame_id && meta.frame_id !== shownFrameId);
  }

  function imageUrl(streamId, frameId) {
    const q = new URLSearchParams({ stream_id: streamId, f: String(frameId || "") });
    return `/api/vision/frames/latest/image?${q.toString()}`;
  }

  const api = { LIVE_FRAME_VIEWS, formatAge, liveFrameState, shouldFetchImage, imageUrl };

  global.OrionVisionLiveFrame = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);

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

  const STALE_BADGE = "bg-red-600/90 text-white";
  const DIM = "opacity-40 grayscale";

  function staleState(text) {
    return { kind: "stale", message: "", badgeText: text, badgeClass: STALE_BADGE, imageClass: DIM };
  }

  // meta: parsed JSON from /api/vision/frames/latest, or null when the
  // fetch failed. picture: what is actually on screen --
  //   { loadedAtMs: when the last image finished loading (null if never),
  //     lastError: true if the most recent image load failed }.
  // LIVE requires BOTH a fresh frame announcement from the backend AND a
  // picture that actually loaded recently -- review finding, 2026-10-15:
  // badging on the backend's freshness alone showed "LIVE" over a blank or
  // frozen image whenever the image route kept failing (missing mount,
  // undecodable file).
  function liveFrameState(meta, view, picture, nowMs) {
    const label = (view && view.label) || "camera";
    const hint = (view && view.hint) || "";
    const pic = picture || {};
    const hasPicture = typeof pic.loadedAtMs === "number";
    if (!meta || !meta.available) {
      if (!hasPicture) {
        const why = meta ? `No frame from ${label} yet` : `Can't reach Hub's ${label} frame feed`;
        return { kind: "none", message: `${why}${hint ? " -- " + hint : "."}`, badgeText: "" };
      }
      // Keep the last picture on a blip, but never call it live.
      return staleState(meta ? "STALE · no frame" : "STALE · connection lost");
    }
    if (meta.stale) {
      return staleState(`STALE · last frame ${formatAge(meta.age_sec)}`);
    }
    if (!hasPicture) {
      if (pic.lastError) return staleState("IMAGE ERROR · frame won't load");
      return {
        kind: "loading",
        message: "",
        badgeText: "LOADING",
        badgeClass: "bg-gray-700/90 text-gray-200",
        imageClass: "",
      };
    }
    // Two poll intervals of slack: carbon only produces a frame every ~5s
    // and is polled every 5s, so its picture is routinely ~10s old.
    const staleMs = (Number(meta.stale_after_sec) || 10) * 1000 + 2 * ((view && view.pollMs) || 0);
    const shownAgeSec = Math.max(0, (nowMs - pic.loadedAtMs) / 1000);
    if (shownAgeSec * 1000 > staleMs) {
      return staleState(
        `STALE · picture ${formatAge(shownAgeSec)}${pic.lastError ? " (image error)" : ""}`
      );
    }
    return {
      kind: "live",
      message: "",
      badgeText: `LIVE · ${formatAge(meta.age_sec)}`,
      badgeClass: "bg-emerald-600/90 text-white",
      imageClass: "",
    };
  }

  // Only pull new image bytes when the backend has a different frame than
  // the one on screen -- the badge still refreshes every poll regardless.
  function shouldFetchImage(meta, shownFrameId) {
    return Boolean(meta && meta.available && meta.frame_id && meta.frame_id !== shownFrameId);
  }

  // ``f`` only busts the browser cache -- the server always returns its
  // newest frame, which may be one newer than frameId.
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

(function (global) {
  // "Walkway history" panel, under the live picture in the Vision panel's
  // "Walkway (live)" view (walkway rollups, 2026-10-18).
  //
  // orion-sql-writer counts the walkway by hour, day, week and month
  // (vision_rollup; orion/vision/walkway_rollup.py): passings by kind, named
  // people, regulars, busiest / quietest, and the hours the camera was down
  // (never shown as "nobody came"). This panel lists those rows with tabs per
  // grain (GET /api/vision/rollups, scripts/walkway_rollup_routes.py).
  //
  // Everything from the database is rendered with textContent, never innerHTML.

  const VIEW_VALUE = "walkway-live";
  const POLL_MS = 120000;
  const GRAINS = [
    { value: "hour", label: "Hours" },
    { value: "day", label: "Days" },
    { value: "week", label: "Weeks" },
    { value: "month", label: "Months" },
  ];
  const CAMERA_TEXT = { ok: "camera up", partial: "camera partly down", down: "camera down", unknown: "camera ?" };

  function el(doc, tag, cls, text) {
    const n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function cameraChip(row) {
    const cam = (row && row.camera) || {};
    const up = Number(cam.up_minutes || 0);
    const expected = Number(cam.expected_minutes || 0);
    const word = CAMERA_TEXT[cam.status] || CAMERA_TEXT.unknown;
    if (!expected) return word;
    if (expected <= 60) return `${word} ${up}/${expected} min`;
    return `${word} ${(up / 60).toFixed(1)}/${(expected / 60).toFixed(1)} h`;
  }

  function captionLines(row) {
    return ((row && row.captions) || [])
      .filter((c) => c && typeof c.text === "string" && c.text.trim())
      .slice(0, 3)
      .map((c) => `caption ${c.at || ""} (one frame, may be wrong): ${c.text}`);
  }

  function renderRow(doc, row) {
    const item = el(doc, "div", "rounded-lg border border-gray-700 bg-gray-800/60 p-2 space-y-1");
    item.setAttribute("data-rollup-label", String(row.label || ""));
    item.appendChild(el(doc, "div", "text-xs text-gray-100 break-words", row.line));
    const meta = [cameraChip(row)];
    if (!row.complete) meta.push(row.grain === "hour" ? "not settled" : "still running or partial");
    const reg = (row.regulars && row.regulars.count) || 0;
    if (reg) meta.push(`${reg} unnamed regular${reg === 1 ? "" : "s"}`);
    item.appendChild(el(doc, "div", "text-[11px] text-gray-400 break-words", meta.join(" · ")));
    captionLines(row).forEach((t) => item.appendChild(el(doc, "div", "text-[11px] italic text-gray-500 break-words", t)));
    return item;
  }

  function mount(doc, fetchFn) {
    const card = doc.getElementById("walkwayHistoryCard");
    const select = doc.getElementById("visionSource");
    if (!card) return null;
    const tabs = doc.getElementById("walkwayHistoryTabs");
    const list = doc.getElementById("walkwayHistoryList");
    const status = doc.getElementById("walkwayHistoryStatus");
    let grain = "day";
    let timer = null;
    const buttons = [];

    function paintTabs() {
      buttons.forEach((b) => {
        const on = b.getAttribute("data-rollup-grain") === grain;
        b.setAttribute("aria-selected", on ? "true" : "false");
        b.className = "min-h-[2rem] rounded-lg border px-3 py-1 text-xs " + (on
          ? "border-sky-400 bg-sky-800 text-white"
          : "border-gray-600 bg-gray-900 text-gray-300 hover:bg-gray-700");
      });
    }

    if (tabs) {
      tabs.textContent = "";
      GRAINS.forEach((g) => {
        const b = el(doc, "button", "", g.label);
        b.setAttribute("type", "button");
        b.setAttribute("role", "tab");
        b.setAttribute("data-rollup-grain", g.value);
        b.addEventListener("click", () => { grain = g.value; paintTabs(); load(); });
        tabs.appendChild(b);
        buttons.push(b);
      });
      paintTabs();
    }

    async function load() {
      const want = grain;
      try {
        const res = await fetchFn(`/api/vision/rollups?stream_id=walkway&grain=${want}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        if (want !== grain) return; // a newer tab click won
        list.textContent = "";
        const rows = body.rows || [];
        rows.forEach((r) => list.appendChild(renderRow(doc, r)));
        if (status) {
          status.textContent = rows.length
            ? `${rows.length} ${want}${rows.length === 1 ? "" : "s"}, newest first. Counted by code from sightings; hours the camera was down say so.`
            : `No ${want} rows yet. The rollup reducer writes one as each hour closes.`;
        }
      } catch (err) {
        if (status) status.textContent = `Can't load walkway history (${err.message}).`;
      }
    }

    function sync() {
      const on = !!select && select.value === VIEW_VALUE;
      card.classList.toggle("hidden", !on);
      if (on && !timer) {
        load();
        timer = setInterval(load, POLL_MS);
      } else if (!on && timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    if (select) select.addEventListener("change", sync);
    sync();
    return { load, sync, setGrain: (g) => { grain = g; paintTabs(); return load(); } };
  }

  const api = { VIEW_VALUE, GRAINS, cameraChip, captionLines, renderRow, mount };

  global.OrionWalkwayHistory = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof document !== "undefined" && typeof global.fetch === "function") {
    const start = function () {
      api.mount(document, global.fetch.bind(global));
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }
})(typeof window !== "undefined" ? window : globalThis);

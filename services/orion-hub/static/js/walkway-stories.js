(function (global) {
  // "Walkway stories" spot-check card, under the live picture in the Vision
  // panel's "Walkway (live)" view (motion stories slice 1, 2026-10-16;
  // docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md).
  //
  // orion-sql-writer turns each passing thing into one sentence from box
  // maths. Juniper keeps them HUB ONLY until she has checked them: >= 50
  // checked, >= 80% fully right, zero wrong names. This card lists recent
  // stories with the pictures they rest on and records Right / Wrong name /
  // Wrong motion / Wrong other (scripts/motion_story_routes.py).
  //
  // Pictures: crop thumbnails exist for only a few boxes (~6%), so every
  // story also gets a box map -- its boxes drawn over today's walkway view,
  // numbered in time order. The view is the latest frame, not the moment.
  //
  // Everything from the database is rendered with textContent / attributes,
  // never innerHTML.

  const VIEW_VALUE = "walkway-live";
  const POLL_MS = 60000;
  const FRAME_W = 2560;
  const FRAME_H = 1920;
  const SCENE_URL = "/api/vision/frames/latest/image?stream_id=walkway";
  const VERDICTS = [
    { value: "right", label: "Right" },
    { value: "wrong_name", label: "Wrong name" },
    { value: "wrong_motion", label: "Wrong motion" },
    { value: "wrong_other", label: "Wrong other" },
  ];
  const THUMB_RE = /^[0-9a-f]{64}$/;
  const SVG_NS = "http://www.w3.org/2000/svg";

  function tallyLine(tally) {
    const t = tally || {};
    const gate = t.gate || { min_checked: 50, min_right: 0.8, max_wrong_names: 0 };
    const checked = Number(t.checked || 0);
    const pct = typeof t.pct_right === "number" ? `${Math.round(t.pct_right * 100)}%` : "–";
    return (
      `Checked ${checked} of ${gate.min_checked} · right ${t.right || 0} (${pct}, need ` +
      `${Math.round(gate.min_right * 100)}%) · wrong names ${t.wrong_name || 0} (must be ${gate.max_wrong_names}) · ` +
      `named stories checked ${t.named_checked || 0}`
    );
  }

  function gateLine(tally) {
    const t = tally || {};
    const gate = t.gate || { min_checked: 50, min_right: 0.8, max_wrong_names: 0 };
    if (gate.passed) return "Gate passed: stories may be switched on for Orion (Juniper decides).";
    const reasons = [];
    const checked = Number(t.checked || 0);
    if (checked < gate.min_checked) reasons.push(`${gate.min_checked - checked} more to check`);
    if (checked && typeof t.pct_right === "number" && t.pct_right < gate.min_right) reasons.push("too many wrong");
    if ((t.wrong_name || 0) > gate.max_wrong_names) reasons.push("a wrong name was found");
    return `Hub only for now: ${reasons.join(", ") || "not yet"}.`;
  }

  // First, middle and last thumbnails that exist, in time order.
  function pickThumbs(evidence, max) {
    const n = max || 3;
    const have = (evidence || []).filter((e) => e && typeof e.thumb === "string" && THUMB_RE.test(e.thumb));
    if (have.length <= n) return have.map((e) => e.thumb);
    const idx = [0, Math.floor((have.length - 1) / 2), have.length - 1];
    return idx.slice(0, n).map((i) => have[i].thumb);
  }

  function boxMapModel(evidence) {
    const boxes = [];
    const feet = [];
    (evidence || []).forEach((e, i) => {
      const b = e && Array.isArray(e.box) ? e.box.map(Number) : null;
      if (!b || b.length !== 4 || b.some((v) => !Number.isFinite(v))) return;
      boxes.push({ i: i + 1, x: b[0], y: b[1], w: Math.max(1, b[2] - b[0]), h: Math.max(1, b[3] - b[1]),
                   clipped: !!e.clipped });
      if (!e.clipped) feet.push(`${Math.round((b[0] + b[2]) / 2)},${Math.round(b[3])}`);
    });
    return { boxes, path: feet.join(" ") };
  }

  function localTime(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso || "");
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function el(doc, tag, cls, text) {
    const n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function svgEl(doc, tag, attrs) {
    const n = doc.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach((k) => n.setAttribute(k, String(attrs[k])));
    return n;
  }

  function renderBoxMap(doc, evidence) {
    const model = boxMapModel(evidence);
    const svg = svgEl(doc, "svg", { viewBox: `0 0 ${FRAME_W} ${FRAME_H}`, role: "img",
      "aria-label": `${model.boxes.length} boxes in time order over today's walkway view`,
      class: "w-full h-auto rounded bg-gray-800", "data-box-map": "1" });
    svg.appendChild(svgEl(doc, "image", { href: SCENE_URL, x: 0, y: 0, width: FRAME_W, height: FRAME_H,
      opacity: 0.55, preserveAspectRatio: "none" }));
    if (model.path) {
      svg.appendChild(svgEl(doc, "polyline", { points: model.path, fill: "none", stroke: "#facc15",
        "stroke-width": 10, "stroke-linejoin": "round" }));
    }
    const last = model.boxes.length;
    model.boxes.forEach((b) => {
      const colour = b.i === 1 ? "#22c55e" : b.i === last ? "#ef4444" : "#60a5fa";
      const r = svgEl(doc, "rect", { x: b.x, y: b.y, width: b.w, height: b.h, fill: "none", stroke: colour,
        "stroke-width": 8 });
      if (b.clipped) r.setAttribute("stroke-dasharray", "30 20");
      svg.appendChild(r);
      const label = svgEl(doc, "text", { x: b.x + 6, y: Math.max(60, b.y - 10), fill: colour, "font-size": 64,
        "font-weight": 700 });
      label.textContent = String(b.i);
      svg.appendChild(label);
    });
    return svg;
  }

  function renderStory(doc, story, onVerdict) {
    const item = el(doc, "div", "rounded-lg border border-gray-700 bg-gray-800/60 p-3 space-y-2");
    item.setAttribute("data-story-id", story.story_id);
    item.appendChild(el(doc, "div", "text-sm text-gray-100", story.sentence));
    const meta = [localTime(story.started_at), `${story.kind}/${story.speed_class || "?"}`,
      `${story.detection_count} detections`];
    if (story.name_label) meta.push("named");
    item.appendChild(el(doc, "div", "text-[11px] text-gray-400", meta.join(" · ")));

    const pics = el(doc, "div", "flex flex-wrap items-start gap-2");
    const mapWrap = el(doc, "div", "w-full max-w-[20rem]");
    mapWrap.appendChild(renderBoxMap(doc, story.evidence));
    mapWrap.appendChild(el(doc, "div", "text-[10px] text-gray-500",
      "Boxes over today's view (not the moment): 1 = first (green), last = red, dashed = cut by the frame edge."));
    pics.appendChild(mapWrap);
    pickThumbs(story.evidence).forEach((h) => {
      const img = el(doc, "img", "h-20 w-auto rounded border border-gray-700");
      img.setAttribute("src", `/api/vision/crop-thumbs/${h}`);
      img.setAttribute("alt", "crop from this story");
      img.setAttribute("loading", "lazy");
      img.setAttribute("data-story-thumb", h);
      // vision-host prunes thumbnails after 10 days: say so instead of a broken image.
      img.addEventListener("error", () => {
        img.replaceWith(el(doc, "span", "text-[10px] text-gray-500", "(picture gone)"));
      });
      pics.appendChild(img);
    });
    item.appendChild(pics);

    const row = el(doc, "div", "flex flex-wrap gap-2");
    VERDICTS.forEach((v) => {
      const btn = el(doc, "button", "", v.label);
      btn.setAttribute("type", "button");
      btn.setAttribute("data-story-verdict", v.value);
      setPressed(btn, story.verdict === v.value);
      btn.addEventListener("click", () => onVerdict(story, v.value, item));
      row.appendChild(btn);
    });
    item.appendChild(row);
    return item;
  }

  function setPressed(btn, on) {
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.className = "min-h-[2.25rem] rounded-lg border px-3 py-1 text-xs " + (on
      ? "border-emerald-400 bg-emerald-700 text-white"
      : "border-gray-600 bg-gray-900 text-gray-200 hover:bg-gray-700");
  }

  function mount(doc, fetchFn) {
    const card = doc.getElementById("walkwayStoriesCard");
    const select = doc.getElementById("visionSource");
    if (!card) return null;
    const list = doc.getElementById("walkwayStoriesList");
    const status = doc.getElementById("walkwayStoriesStatus");
    const tallyEl = doc.getElementById("walkwayStoriesTally");
    const gateEl = doc.getElementById("walkwayStoriesGate");
    const uncheckedBox = doc.getElementById("walkwayStoriesUnchecked");
    let timer = null;

    function showTally(t) {
      if (tallyEl) tallyEl.textContent = tallyLine(t);
      if (gateEl) gateEl.textContent = gateLine(t);
    }

    async function onVerdict(story, verdict, item) {
      const buttons = item.querySelectorAll("[data-story-verdict]");
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetchFn(`/api/vision/motion-stories/${encodeURIComponent(story.story_id)}/verdict`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ verdict }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        story.verdict = verdict;
        buttons.forEach((b) => setPressed(b, b.getAttribute("data-story-verdict") === verdict));
        showTally(body.tally);
        if (status) status.textContent = "Saved.";
      } catch (err) {
        if (status) status.textContent = `Could not save that verdict (${err.message}). Try again.`;
      } finally {
        buttons.forEach((b) => { b.disabled = false; });
      }
    }

    async function load() {
      const unchecked = uncheckedBox && uncheckedBox.checked ? "true" : "false";
      try {
        const res = await fetchFn(`/api/vision/motion-stories?limit=20&unchecked_only=${unchecked}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        showTally(body.tally);
        list.textContent = "";
        const stories = body.stories || [];
        stories.forEach((s) => list.appendChild(renderStory(doc, s, onVerdict)));
        if (status) {
          status.textContent = stories.length
            ? `${stories.length} recent ${stories.length === 1 ? "story" : "stories"} (clean frames only).`
            : "No stories yet. Someone has to walk past first (walkers only: runners, bikes and cars are too fast for the camera today).";
        }
      } catch (err) {
        if (status) status.textContent = `Can't load walkway stories (${err.message}).`;
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
    if (uncheckedBox) uncheckedBox.addEventListener("change", load);
    sync();
    return { load, sync };
  }

  const api = { VIEW_VALUE, VERDICTS, tallyLine, gateLine, pickThumbs, boxMapModel, renderStory, mount };

  global.OrionWalkwayStories = api;
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

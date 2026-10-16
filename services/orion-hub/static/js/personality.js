/* Personality panel (templates/personality.html, scripts/personality_routes.py).
 *
 * Read-only observability for Orion's personality graph (spec rev 2, section 7). Pure model
 * functions are exported for node:test; the DOM layer below renders only what they return, and
 * writes every piece of upstream text with textContent (labels and excerpts are Orion's raw words).
 * The one innerHTML write is the sparkline, whose markup is numbers plus an escaped title.
 * Data comes from:
 *   GET /api/personality/cue                          the frame + the cue text the lanes get
 *   GET /api/personality/dispositions?status=S&limit=N nodes (+ history) per status
 *   GET /api/personality/evidence?disposition_id=&limit=N one node's evidence rows
 */
(function (root) {
  "use strict";

  const NODE_LIMIT = 50;
  const EVIDENCE_LIMIT = 50;
  const EMPTY_TEXT = "The graph is empty — it starts from today and grows as Orion writes.";

  const SKIP_TEXT = {
    disabled: "The cue is turned off.",
    fetch_failed: "The lanes could not fetch the frame, so they get no cue.",
    invalid_frame: "The frame did not validate, so the lanes get no cue.",
    stale: "The frame is older than an hour, so the lanes get no cue.",
    empty: "No active node and no curiosity choices yet, so the lanes get no cue.",
    render_failed: "The cue renderer failed, so the lanes get no cue.",
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmtAt(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function fmtNum(value, digits) {
    if (value == null || value === "") return "—";
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return n.toFixed(digits);
  }

  function fmtSigned(value, digits) {
    const text = fmtNum(value, digits);
    if (text === "—") return text;
    return Number(value) > 0 ? `+${text}` : text;
  }

  // "dream: 3, journal: 1" -- most first, ties by name, zero counts dropped.
  function lanesText(laneCounts) {
    if (!laneCounts || typeof laneCounts !== "object") return "—";
    const parts = Object.entries(laneCounts)
      .map(([lane, n]) => [lane, Number(n)])
      .filter(([, n]) => Number.isFinite(n) && n > 0)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([lane, n]) => `${lane}: ${n}`);
    return parts.length ? parts.join(", ") : "—";
  }

  // Weight history -> polyline points in a width x height box, oldest left. Null when no point
  // has a finite weight. One point draws a flat line; a flat history sits mid-box.
  function sparklineModel(history, width, height) {
    const w = width || 80;
    const h = height || 18;
    const pts = (Array.isArray(history) ? history : [])
      .filter((p) => p && Number.isFinite(Number(p.weight)))
      .map((p) => ({ t: Date.parse(p.at), at: String(p.at || ""), weight: Number(p.weight) }))
      .sort((a, b) => (Number.isFinite(a.t) && Number.isFinite(b.t) ? a.t - b.t : a.at.localeCompare(b.at)));
    if (!pts.length) return null;
    const ws = pts.map((p) => p.weight);
    const min = Math.min(...ws);
    const max = Math.max(...ws);
    const span = max - min;
    const pad = 1;
    const y = (v) => (span > 0 ? pad + (h - 2 * pad) * (1 - (v - min) / span) : h / 2);
    const series = pts.length === 1 ? [pts[0], pts[0]] : pts;
    const step = (w - 2 * pad) / (series.length - 1);
    const points = series.map((p, i) => `${(pad + i * step).toFixed(1)},${y(p.weight).toFixed(1)}`).join(" ");
    return { points, width: w, height: h, n: pts.length, min, max,
             first: pts[0].weight, last: pts[pts.length - 1].weight };
  }

  function sparklineSvg(history, width, height) {
    const m = sparklineModel(history, width, height);
    if (!m) return "";
    const title = `weight ${fmtNum(m.first, 3)} → ${fmtNum(m.last, 3)} over ${m.n} point${m.n === 1 ? "" : "s"}`;
    return `<svg class="spark" width="${m.width}" height="${m.height}" viewBox="0 0 ${m.width} ${m.height}" ` +
      `role="img" aria-label="${escapeHtml(title)}"><title>${escapeHtml(title)}</title>` +
      `<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="${m.points}"/></svg>`;
  }

  // Upstream returns either a bare list or an object wrapping it.
  function listFrom(body, keys) {
    if (Array.isArray(body)) return body;
    if (body && typeof body === "object") {
      for (const k of keys) if (Array.isArray(body[k])) return body[k];
    }
    return [];
  }
  const nodesFrom = (body) => listFrom(body, ["dispositions", "nodes", "items"]);
  const evidenceFrom = (body) => listFrom(body, ["evidence", "rows", "items"]);

  function dispositionRow(d) {
    const node = d || {};
    return {
      id: String(node.disposition_id || ""),
      label: String(node.label || "(no label)"),
      status: String(node.status || "unknown"),
      origin: String(node.origin || ""),
      weight: fmtNum(node.weight, 3),
      evidence: Number.isFinite(Number(node.evidence_count)) ? Number(node.evidence_count) : 0,
      lanes: lanesText(node.lane_counts),
      chosen: Number.isFinite(Number(node.chosen_count)) ? Number(node.chosen_count) : 0,
      first_seen: fmtAt(node.first_seen_at),
      last_reinforced: fmtAt(node.last_reinforced_at),
      history: Array.isArray(node.history) ? node.history : [],
    };
  }

  function evidenceRow(e) {
    const row = e || {};
    return {
      id: String(row.evidence_id || ""),
      lane: String(row.lane || "?"),
      at: fmtAt(row.occurred_at),
      excerpt: String(row.excerpt || ""),
      similarity: fmtNum(row.similarity, 2),
      weight_delta: fmtSigned(row.weight_delta, 3),
      source_ref: String(row.source_ref || ""),
    };
  }

  // What the page leads with, from the /api/personality/cue response.
  function pageState(resp) {
    const r = resp || {};
    const frame = r.frame && typeof r.frame === "object" ? r.frame : null;
    if (!frame) {
      const why = r.error || "no frame";
      return { kind: "unavailable", message: `durable-runs unavailable: ${why}` };
    }
    // A frame that did not validate is not an empty graph: never read it as calm.
    if (r.cue && r.cue.skip_reason === "invalid_frame") {
      return { kind: "unavailable", message: "durable-runs sent a frame that does not validate (PersonalityFrameV1)." };
    }
    const total = ["active_count", "candidate_count", "retired_count", "evidence_rows"]
      .reduce((acc, k) => acc + (Number(frame[k]) || 0), 0);
    if (total === 0 && !(Array.isArray(frame.dispositions) && frame.dispositions.length)) {
      return { kind: "empty", message: EMPTY_TEXT };
    }
    return { kind: "ok", message: "" };
  }

  function countsText(frame) {
    const f = frame || {};
    const n = (k) => (Number.isFinite(Number(f[k])) ? Number(f[k]) : 0);
    return `${n("active_count")} active · ${n("candidate_count")} candidate · ${n("retired_count")} retired · ` +
      `${n("evidence_rows")} evidence rows`;
  }

  // "(12)", or "(50 of 80)" when the list is capped below the frame's own count.
  function shownCountText(shown, total) {
    const t = Number(total);
    return Number.isFinite(t) && t > shown ? `(${shown} of ${t})` : `(${shown})`;
  }

  function renderParamsText(params) {
    if (!params) return "";
    return `rendered with max age ${params.max_age_sec}s, top ${params.top_k} — the lanes' default settings`;
  }

  function choiceStyleText(frame) {
    const s = frame && frame.choice_style;
    if (!s || !(Number(s.choices) > 0)) return "No curiosity choices recorded yet.";
    return `In curiosity over the last ${s.window_days} days: took the top suggestion ${s.took_top} of ` +
      `${s.choices} times, went their own way ${s.went_own_way} of ${s.choices}.`;
  }

  function skipText(cue) {
    if (!cue || !cue.skip_reason) return "";
    return `${cue.skip_reason}: ${SKIP_TEXT[cue.skip_reason] || "no cue rendered."}`;
  }

  const api = {
    escapeHtml, fmtAt, fmtNum, fmtSigned, lanesText, sparklineModel, sparklineSvg, nodesFrom, evidenceFrom,
    dispositionRow, evidenceRow, pageState, countsText, shownCountText, renderParamsText, choiceStyleText, skipText, EMPTY_TEXT, NODE_LIMIT,
    EVIDENCE_LIMIT,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // --- DOM layer ------------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const view = { seq: 0, frame: null };
  const COLUMNS = ["Label", "Weight", "Evidence", "Lanes", "Chosen", "First seen", "Last reinforced"];
  const NUM_COLUMNS = new Set(["Weight", "Evidence", "Chosen"]);

  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === "class") node.className = v; else node.setAttribute(k, v);
    });
    if (text != null) node.textContent = String(text);
    return node;
  }

  async function getJson(url) {
    const resp = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
    let body = null;
    try { body = await resp.json(); } catch (_) { body = null; }
    if (!resp.ok) {
      const detail = body && body.detail ? body.detail : `HTTP ${resp.status}`;
      const err = new Error(String(detail));
      err.status = resp.status;
      throw err;
    }
    return body;
  }

  function setStatus(text, kind) {
    const node = $("pStatus");
    node.textContent = text || "";
    node.classList.toggle("err", kind === "unavailable" || kind === "error");
  }

  function renderCue(resp) {
    const cue = (resp && resp.cue) || {};
    const pre = $("pCueText");
    pre.textContent = cue.text || "";
    pre.hidden = !cue.text;
    $("pCueSkip").textContent = skipText(cue) + (resp && resp.error ? ` (${resp.error})` : "");
    $("pCueMeta").textContent = cue.rendered_at ? `rendered ${fmtAt(cue.rendered_at)}` : "";
    $("pCueParams").textContent = renderParamsText(resp && resp.render_params);
  }

  function renderFrame(frame) {
    $("pCounts").textContent = frame ? countsText(frame) : "—";
    $("pChoiceStyle").textContent = frame ? choiceStyleText(frame) : "—";
    $("pGeneratedAt").textContent = frame && frame.generated_at ? `frame generated ${fmtAt(frame.generated_at)}` : "";
    const warnings = frame && Array.isArray(frame.warnings) ? frame.warnings : [];
    $("pWarnings").textContent = warnings.length ? `warnings: ${warnings.join(" · ")}` : "";
  }

  function renderEvidence(cell, rows) {
    cell.replaceChildren();
    if (!rows.length) { cell.append(el("div", { class: "muted" }, "No evidence rows for this node.")); return; }
    const table = el("table", { class: "evidence" });
    const head = el("tr");
    ["Lane", "When", "Excerpt", "Similarity", "Weight Δ"].forEach((h) => head.append(el("th", null, h)));
    table.append(head);
    rows.map(evidenceRow).forEach((r) => {
      const tr = el("tr");
      tr.append(el("td", null, r.lane));
      tr.append(el("td", { class: "time" }, r.at));
      tr.append(el("td", { class: "excerpt", title: r.source_ref }, r.excerpt));
      tr.append(el("td", { class: "num" }, r.similarity));
      tr.append(el("td", { class: "num" }, r.weight_delta));
      table.append(tr);
    });
    cell.append(table);
  }

  async function toggleEvidence(tr, row, button) {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("evidence-row")) {
      next.remove();
      tr.classList.remove("open");
      button.setAttribute("aria-expanded", "false");
      return;
    }
    tr.classList.add("open");
    button.setAttribute("aria-expanded", "true");
    const holder = el("tr", { class: "evidence-row", id: button.getAttribute("aria-controls") });
    const cell = el("td", { colspan: String(COLUMNS.length) });
    cell.append(el("div", { class: "muted" }, "loading evidence…"));
    holder.append(cell);
    tr.after(holder);
    try {
      const body = await getJson(`/api/personality/evidence?disposition_id=${encodeURIComponent(row.id)}&limit=${EVIDENCE_LIMIT}`);
      if (!holder.isConnected) return;   // closed (or refreshed away) while loading
      renderEvidence(cell, evidenceFrom(body));
    } catch (err) {
      if (holder.isConnected) cell.replaceChildren(el("div", { class: "err" }, `Could not load evidence: ${err.message}`));
    }
  }

  function renderNodes(boxId, countId, nodes, emptyText, total) {
    const box = $(boxId);
    box.replaceChildren();
    if (countId) $(countId).textContent = shownCountText(nodes.length, total);
    if (!nodes.length) { box.append(el("div", { class: "muted" }, emptyText)); return; }
    const scroll = el("div", { class: "scroll" });
    const table = el("table");
    const head = el("tr");
    COLUMNS.forEach((h) => head.append(el("th", NUM_COLUMNS.has(h) ? { class: "num" } : null, h)));
    table.append(head);
    nodes.map(dispositionRow).forEach((r, i) => {
      const tr = el("tr", { class: "node", title: `${r.origin} · ${r.id} — click for evidence` });
      // The label is a real button (keyboard + screen reader); the whole row is also clickable.
      const button = el("button", { type: "button", class: "label-btn", "aria-expanded": "false",
                                    "aria-controls": `${boxId}-ev-${i}` }, r.label);
      const labelCell = el("td", { class: "label" });
      labelCell.append(button);
      tr.append(labelCell);
      const w = el("td", { class: "num weight" });
      const spark = el("span", { class: "spark-wrap" });
      spark.innerHTML = sparklineSvg(r.history);   // numbers + escaped title only
      w.append(spark, " ", el("span", null, r.weight));
      tr.append(w);
      tr.append(el("td", { class: "num" }, r.evidence));
      tr.append(el("td", { class: "lanes" }, r.lanes));
      tr.append(el("td", { class: "num" }, r.chosen));
      tr.append(el("td", { class: "time" }, r.first_seen));
      tr.append(el("td", { class: "time" }, r.last_reinforced));
      tr.addEventListener("click", () => toggleEvidence(tr, r, button));
      table.append(tr);
    });
    scroll.append(table);
    box.append(scroll);
  }

  async function loadNodes(status, seq) {
    const target = { active: ["pActive", "pActiveCount", "No active nodes yet."],
                     candidate: ["pCandidates", "pCandidateCount", "No candidates."],
                     retired: ["pRetired", "pRetiredCount", "Nothing retired."] }[status];
    try {
      const body = await getJson(`/api/personality/dispositions?status=${status}&limit=${NODE_LIMIT}`);
      if (seq !== view.seq) return;
      renderNodes(target[0], target[1], nodesFrom(body), target[2], (view.frame || {})[`${status}_count`]);
    } catch (err) {
      if (seq !== view.seq) return;
      $(target[1]).textContent = "";
      $(target[0]).replaceChildren(el("div", { class: "err" }, `Could not load ${status} nodes: ${err.message}`));
    }
  }

  function clearNodes() {
    ["pActive", "pCandidates", "pRetired"].forEach((id) => $(id).replaceChildren());
    ["pActiveCount", "pCandidateCount", "pRetiredCount"].forEach((id) => { $(id).textContent = ""; });
  }

  // A Hub-side failure must not leave the previous cue and tables looking current.
  function clearAll() {
    view.frame = null;
    renderCue(null);
    renderFrame(null);
    clearNodes();
  }

  async function refresh() {
    const seq = ++view.seq;
    setStatus("loading…");
    let resp;
    try {
      resp = await getJson("/api/personality/cue");
    } catch (err) {
      if (seq !== view.seq) return;
      setStatus(`Hub could not render the cue: ${err.message}`, "error");
      clearAll();
      return;
    }
    if (seq !== view.seq) return;
    view.frame = resp.frame || null;
    renderCue(resp);
    renderFrame(resp.frame);
    const state = pageState(resp);
    setStatus(state.message, state.kind);
    if (state.kind === "unavailable") {
      clearNodes();
      return;
    }
    await Promise.all(["active", "candidate", "retired"].map((s) => loadNodes(s, seq)));
  }

  // Inside the Hub tab iframe, document.hidden only tracks the browser tab, not the Hub panel's
  // display:none -- a hidden panel gives the frame element no offsetParent.
  function isVisible() {
    if (document.hidden) return false;
    try {
      const host = root.frameElement;
      if (host && host.offsetParent === null) return false;
    } catch (_) { /* cross-origin: assume visible */ }
    return true;
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("pRefresh").addEventListener("click", refresh);
    refresh();
    // Skip the poll while an evidence list is open so it does not collapse under the reader.
    setInterval(() => { if (isVisible() && !document.querySelector("tr.evidence-row")) refresh(); }, 60000);
  });
})(typeof window !== "undefined" ? window : globalThis);

/* Temporal Self day view (templates/temporal_self.html, scripts/temporal_self_routes.py).
 *
 * Read-only. Pure model functions (timelineModel, regulationModel, shiftDay, ...) are exported
 * for node:test; the DOM layer below only renders what they return. Data comes from:
 *   GET /api/temporal-self/frame          current day frame (day phase, active arc, threads)
 *   GET /api/temporal-self/arcs?limit=N   every arc of the current day (frame.arcs_today is capped)
 *   GET /api/temporal-self/day/{day_id}   a closed day: final frame + every arc (404 = not closed)
 *   GET /api/temporal-self/regulation     arousal level + since, drives (rest)
 *   GET /api/temporal-self/immune         immune findings + chronicle health
 */
(function (root) {
  "use strict";

  const ARC_LIMIT = 2000;

  function fmtDuration(sec) {
    if (sec == null || sec === "") return "—";
    const s = Number(sec);
    if (!Number.isFinite(s) || s < 0) return "—";
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${Math.round(s / 60)}m`;
    const h = Math.floor(s / 3600);
    const m = Math.round((s % 3600) / 60);
    return m ? `${h}h ${m}m` : `${h}h`;
  }

  function fmtTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function fmtAt(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  // A label is often empty upstream (imagery, concern loops); fall back to the subject ref.
  function arcLabel(arc) {
    const label = String((arc && arc.subject_label) || "").trim();
    if (label) return label;
    return String((arc && arc.subject_ref) || "(no subject)");
  }

  // A time on the given day reads as HH:MM; one on another day (carried arcs) gets its date.
  function fmtOnDay(iso, dayId) {
    if (!iso) return "";
    if (dayId && String(iso).slice(0, 10) < dayId) return fmtAt(iso);
    return fmtTime(iso);
  }

  function arcRow(arc, dayId) {
    const open = (arc.status || "") === "open";
    return {
      arc_id: arc.arc_id || "",
      kind: arc.kind || "unknown",
      label: arcLabel(arc),
      began_at: arc.began_at || null,
      ended_at: arc.ended_at || null,
      span: `${fmtOnDay(arc.began_at, dayId)}–${open ? "now" : fmtTime(arc.ended_at)}`,
      dwell: fmtDuration(arc.cumulative_dwell_sec),
      dwell_sec: Number(arc.cumulative_dwell_sec) || 0,
      returns: Number(arc.attention_returns) || 0,
      status: open ? "open" : (arc.status || "closed"),
      carried: Boolean(arc.carried_from_previous_day),
    };
  }

  // Groups arcs by kind (biggest group first), each group oldest first.
  function timelineModel(arcs, dayId) {
    const groups = new Map();
    (Array.isArray(arcs) ? arcs : []).forEach((arc) => {
      if (!arc || typeof arc !== "object") return;
      const row = arcRow(arc, dayId);
      if (!groups.has(row.kind)) groups.set(row.kind, []);
      groups.get(row.kind).push(row);
    });
    const out = [];
    groups.forEach((rows, kind) => {
      rows.sort((a, b) => String(a.began_at || "").localeCompare(String(b.began_at || "")));
      out.push({
        kind,
        rows,
        count: rows.length,
        open: rows.filter((r) => r.status === "open").length,
        dwell: fmtDuration(rows.reduce((acc, r) => acc + r.dwell_sec, 0)),
      });
    });
    out.sort((a, b) => b.count - a.count || a.kind.localeCompare(b.kind));
    return out;
  }

  function regulationModel(reg, immune) {
    const arousal = (reg && reg.arousal) || {};
    const drives = (reg && Array.isArray(reg.drives)) ? reg.drives : [];
    const rest = drives.find((d) => d && d.drive === "rest") || null;
    const im = (immune && immune.immune) || null;
    const findings = im && Array.isArray(im.findings) ? im.findings : [];
    let openCount = null;
    if (im && !im.last_run_at) openCount = null;   // never ran: a 0 here would read as calm
    else if (im && im.open != null && Number.isFinite(Number(im.open))) openCount = Number(im.open);
    else if (reg && Number.isFinite(Number(reg.immune_findings_open))) openCount = Number(reg.immune_findings_open);
    return {
      arousal_level: arousal.arousal_level || "unknown",
      since: arousal.since || null,
      reasons: Array.isArray(arousal.reasons) ? arousal.reasons : [],
      rest: rest ? {
        level: rest.level,
        threshold: rest.threshold,
        state: rest.state || "unknown",
        last_discharge_at: rest.last_discharge_at || null,
        no_reading_reason: rest.no_reading_reason || null,
      } : null,
      immune_open: openCount,
      findings,
      immune_last_run_at: im ? im.last_run_at || null : null,
      immune_never_ran: Boolean(im && !im.last_run_at),
      immune_skipped: im ? im.skipped || null : null,
      immune_error: im ? im.error || null : null,
      warnings: (reg && Array.isArray(reg.warnings)) ? reg.warnings : [],
    };
  }

  // YYYY-MM-DD +/- days, in UTC arithmetic (day ids are calendar labels, not instants).
  function shiftDay(dayId, delta) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(dayId || ""));
    if (!m) return null;
    const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]) + delta));
    return d.toISOString().slice(0, 10);
  }

  function findingText(f) {
    if (f == null) return "";
    if (typeof f === "string") return f;
    return String(f.summary || f.message || f.kind || f.code || JSON.stringify(f));
  }

  const api = { fmtDuration, fmtOnDay, arcLabel, arcRow, timelineModel, regulationModel, shiftDay, findingText, ARC_LIMIT };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // --- DOM layer ------------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const view = { today: null, dayId: null, seq: 0 };

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

  function setStatus(text, isError) {
    const node = $("tsStatus");
    node.textContent = text || "";
    node.classList.toggle("err", Boolean(isError));
  }

  function renderArcCell(arc) {
    if (!arc) return el("span", { class: "muted" }, "none");
    const row = arcRow(arc);
    const wrap = el("div");
    wrap.append(el("span", { class: `badge kind` }, row.kind), " ", el("strong", null, row.label));
    wrap.append(el("div", { class: "muted" }, `${row.span} · dwell ${row.dwell} · returns ${row.returns}`));
    return wrap;
  }

  function renderThreads(threads) {
    const box = $("tsThreads");
    box.replaceChildren();
    const list = Array.isArray(threads) ? threads : [];
    if (!list.length) { box.append(el("div", { class: "muted" }, "No open threads.")); return; }
    list.forEach((t) => {
      const item = el("div", { class: "thread" });
      item.append(el("strong", null, t.subject_label || t.subject_ref || "(no subject)"));
      item.append(el("div", { class: "muted" },
        `returns today ${t.returns_today ?? 0} · last ${fmtAt(t.last_returned)}` +
        (t.carried_from_previous_day ? " · carried from yesterday" : "")));
      box.append(item);
    });
  }

  function renderTimeline(arcs, total, dayId) {
    const box = $("tsTimeline");
    box.replaceChildren();
    const groups = timelineModel(arcs, dayId);
    const shown = groups.reduce((acc, g) => acc + g.count, 0);
    $("tsArcCount").textContent = total && total > shown ? `${shown} of ${total} arcs` : `${shown} arcs`;
    if (!groups.length) { box.append(el("div", { class: "muted" }, "No arcs for this day.")); return; }
    groups.forEach((g) => {
      const details = el("details", { class: "kind-group" });
      if (groups.length <= 3 || g.open) details.open = true;
      details.append(el("summary", null, `${g.kind} — ${g.count} arcs · ${g.open} open · dwell ${g.dwell}`));
      const scroll = el("div", { class: "scroll" });
      const table = el("table");
      const head = el("tr");
      ["Time", "Subject", "Dwell", "Returns", "Status"].forEach((h) => head.append(el("th", null, h)));
      table.append(head);
      g.rows.forEach((r) => {
        const tr = el("tr", { class: `st-${r.status}` });
        tr.append(el("td", { class: "time" }, r.span));
        tr.append(el("td", { class: "label", title: r.arc_id }, r.label + (r.carried ? " ↩" : "")));
        tr.append(el("td", { class: "num" }, r.dwell));
        tr.append(el("td", { class: "num" }, r.returns));
        tr.append(el("td", null, r.status));
        table.append(tr);
      });
      scroll.append(table);
      details.append(scroll);
      box.append(details);
    });
  }

  function renderRegulation(model) {
    $("tsArousal").textContent = model.arousal_level;
    $("tsArousalSince").textContent = model.since ? `since ${fmtAt(model.since)}` : "";
    $("tsArousalReasons").textContent = model.reasons.join(", ");
    $("tsRest").textContent = model.rest
      ? `${model.rest.state} · level ${model.rest.level ?? "?"} / ${model.rest.threshold ?? "?"}` +
        (model.rest.last_discharge_at ? ` · last slept ${fmtAt(model.rest.last_discharge_at)}` : "")
      : "no rest reading";
    $("tsImmuneCount").textContent = model.immune_open != null ? String(model.immune_open)
      : (model.immune_never_ran ? "not run yet" : "?");
    const meta = [];
    if (model.immune_last_run_at) meta.push(`last run ${fmtAt(model.immune_last_run_at)}`);
    if (model.immune_skipped) meta.push(`skipped: ${model.immune_skipped}`);
    if (model.immune_error) meta.push(`error: ${model.immune_error}`);
    $("tsImmuneMeta").textContent = meta.join(" · ");
    const list = $("tsImmuneFindings");
    list.replaceChildren();
    model.findings.forEach((f) => list.append(el("li", null, findingText(f))));
  }

  async function loadRegulation() {
    const [reg, imm] = await Promise.allSettled([getJson("/api/temporal-self/regulation"),
                                                 getJson("/api/temporal-self/immune")]);
    if (reg.status === "rejected" && imm.status === "rejected") {
      $("tsArousal").textContent = "unavailable";
      $("tsArousalSince").textContent = reg.reason.message;
      return;
    }
    renderRegulation(regulationModel(reg.status === "fulfilled" ? reg.value : null,
                                     imm.status === "fulfilled" ? imm.value : null));
  }

  function renderFrame(frame, isToday, closedAt) {
    $("tsDayId").textContent = frame.day_id || "?";
    $("tsDayPhase").textContent = frame.day_phase || "?";
    $("tsAsOf").textContent = isToday ? `as of ${fmtAt(frame.as_of)}` : `closed ${fmtAt(closedAt || frame.as_of)}`;
    const active = $("tsActiveArc");
    active.replaceChildren(renderArcCell(isToday ? frame.active_arc : null));
    $("tsActiveLabel").textContent = isToday ? "Active arc" : "Active arc (none on a closed day)";
    renderThreads(frame.open_threads);
    const warnings = Array.isArray(frame.warnings) ? frame.warnings : [];
    $("tsWarnings").textContent = warnings.join(" · ");
  }

  // Every load takes a ticket; a load whose ticket is no longer the latest drops its result, so a
  // slow poll of today cannot overwrite a past day the user navigated to meanwhile.
  async function loadToday() {
    const seq = ++view.seq;
    setStatus("loading…");
    try {
      const frame = await getJson("/api/temporal-self/frame");
      if (seq !== view.seq) return;
      view.today = frame.day_id;
      view.dayId = frame.day_id;
      $("tsDayPicker").value = frame.day_id;
      $("tsDayPicker").max = frame.day_id;
      renderFrame(frame, true);
      const arcs = await getJson(`/api/temporal-self/arcs?day_id=${encodeURIComponent(frame.day_id)}&limit=${ARC_LIMIT}`);
      if (seq !== view.seq) return;
      renderTimeline(arcs.arcs, frame.arcs_today_total, frame.day_id);
      setStatus("");
    } catch (err) {
      if (seq !== view.seq) return;
      setStatus(`Could not load today: ${err.message}`, true);
    }
  }

  async function loadDay(dayId) {
    if (!dayId) return;
    if (view.today && dayId >= view.today) { loadToday(); return; }
    const seq = ++view.seq;
    setStatus(`loading ${dayId}…`);
    try {
      const day = await getJson(`/api/temporal-self/day/${encodeURIComponent(dayId)}`);
      if (seq !== view.seq) return;
      view.dayId = dayId;
      $("tsDayPicker").value = dayId;
      renderFrame(day.frame || { day_id: dayId, as_of: day.closed_at }, false, day.closed_at);
      renderTimeline(day.arcs, (day.frame || {}).arcs_today_total, dayId);
      setStatus("");
    } catch (err) {
      if (seq !== view.seq) return;
      $("tsDayPicker").value = view.dayId || view.today || "";
      setStatus(err.status === 404 ? `${dayId}: not closed (or not chronicled)` : `Could not load ${dayId}: ${err.message}`, true);
    }
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

  function refresh() {
    loadRegulation();
    if (!view.dayId || view.dayId === view.today) loadToday(); else loadDay(view.dayId);
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("tsToday").addEventListener("click", () => { view.dayId = null; loadToday(); });
    $("tsPrevDay").addEventListener("click", () => loadDay(shiftDay(view.dayId || view.today, -1)));
    $("tsNextDay").addEventListener("click", () => loadDay(shiftDay(view.dayId || view.today, 1)));
    $("tsDayPicker").addEventListener("change", () => loadDay($("tsDayPicker").value));
    $("tsRefresh").addEventListener("click", refresh);
    refresh();
    setInterval(() => { if (isVisible() && (!view.dayId || view.dayId === view.today)) refresh(); }, 60000);
  });
})(typeof window !== "undefined" ? window : globalThis);

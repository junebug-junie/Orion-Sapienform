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
 *   GET /api/personality/naming?disposition_id=        one node's self-naming log (every attempt)
 * Names, name sentences and refusal reasons are model-written: textContent only, like labels.
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

  // The situations a node's evidence happened in (durable-runs ``situation_mix``, stamped on
  // evidence since 2026-10-16; older rows carry none). Per dimension, the most common value and its
  // share of the n stamped rows: "idle 70% · night 55% · alone 80% (n=20)". A tie shows every tied
  // value, by name ("engaged/idle 50%"). Plain text: the DOM layer writes it with textContent.
  const NO_SITUATION_TEXT = "no situation data yet";

  function topShare(counts, n) {
    const entries = Object.entries(counts && typeof counts === "object" ? counts : {})
      .map(([k, c]) => [k, Number(c)])
      .filter(([, c]) => Number.isFinite(c) && c > 0);
    if (!entries.length) return null;
    const best = Math.max(...entries.map(([, c]) => c));
    const names = entries.filter(([, c]) => c === best).map(([k]) => k).sort((a, b) => a.localeCompare(b));
    return `${names.join("/")} ${Math.round((100 * best) / n)}%`;
  }

  function situationParts(mix) {
    const m = mix && typeof mix === "object" ? mix : {};
    const n = Number(m.n);
    if (!Number.isFinite(n) || n <= 0) return null;
    const company = m.company && typeof m.company === "object" ? m.company : {};
    const companyNamed = { alone: company.false, "with company": company.true };
    // Company share is over rows where company is KNOWN: rows past a stalled chronicle are
    // stamped company=null and must not read as "alone".
    const known = [company.true, company.false].map(Number).filter((c) => Number.isFinite(c) && c > 0)
      .reduce((a, b) => a + b, 0);
    return { n, parts: [topShare(m.arousal, n), topShare(m.day_phase, n), known ? topShare(companyNamed, known) : null]
      .filter(Boolean) };
  }

  function situationText(mix) {
    const s = situationParts(mix);
    if (!s) return NO_SITUATION_TEXT;
    return `${s.parts.length ? s.parts.join(" · ") + " " : ""}(n=${s.n})`;
  }

  // Full breakdown for the cell's tooltip: "arousal: idle 14, engaged 6 | day phase: ...".
  function situationDetail(mix) {
    const s = situationParts(mix);
    if (!s) return NO_SITUATION_TEXT;
    const list = (counts) => Object.entries(counts && typeof counts === "object" ? counts : {})
      .map(([k, c]) => [k, Number(c)]).filter(([, c]) => Number.isFinite(c) && c > 0)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k, c]) => `${k} ${c}`).join(", ") || "—";
    const c = mix.company || {};
    return `arousal: ${list(mix.arousal)} | day phase: ${list(mix.day_phase)} | ` +
      `company: ${list({ alone: c.false, "with company": c.true })} | n=${s.n} rows with a situation`;
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
      when: situationText(node.situation_mix),
      when_detail: situationDetail(node.situation_mix),
    };
  }

  // --- self-naming (spec 2026-10-17 section 6) ---------------------------------------------------
  // The verdict badge for a node. Rows from a durable-runs older than self-naming carry none of
  // the naming keys: that reads "naming not reported", never "not yet shown".
  const NAMING_KEYS = ["name", "orion_verdict", "merged_into", "name_grounding_rank"];

  function nameView(node, lookup) {
    const n = node || {};
    const name = typeof n.name === "string" ? n.name.trim() : "";
    const sentence = typeof n.name_sentence === "string" ? n.name_sentence.trim() : "";
    const rank = Number(n.name_grounding_rank);
    const margin = Number(n.name_grounding_margin);
    const grounding = name && Number.isFinite(rank) && rank > 1
      ? `points elsewhere: rank ${rank}${Number.isFinite(margin) ? ` (margin ${margin.toFixed(2)})` : ""}` : "";
    let badge;
    // 'rejected' is the column value durable-runs was asked for; 'reject' is the NamingVerdict
    // literal (orion/schemas/personality.py). Either is Orion saying "not me".
    if (n.orion_verdict === "rejected" || n.orion_verdict === "reject") badge = { kind: "rejected", text: "Orion: not me" };
    else if (n.merged_into) {
      const target = lookup && typeof lookup === "function" ? lookup(String(n.merged_into)) : null;
      badge = { kind: "merged", text: `merged into ${target || n.merged_into}` };
    } else if (name) badge = { kind: "named", text: "named" };
    else if (NAMING_KEYS.some((k) => Object.prototype.hasOwnProperty.call(n, k))) {
      badge = { kind: "unshown", text: "not yet shown" };
    } else badge = { kind: "unknown", text: "naming not reported" };
    return { name, sentence, grounding, badge };
  }

  const namingFrom = (body) => listFrom(body, ["naming", "rows", "items"]);

  function asList(value) {
    if (Array.isArray(value)) return value.map((v) => String(v));
    if (value == null || value === "") return [];
    if (typeof value === "string") {
      try { const parsed = JSON.parse(value); if (Array.isArray(parsed)) return parsed.map((v) => String(v)); }
      catch (_) { /* plain text */ }
      return [value];
    }
    return [String(value)];
  }

  // The packet Orion was shown, pretty-printed; jsonb may arrive as an object or a JSON string.
  function packetText(shown) {
    if (shown == null || shown === "") return "";
    let v = shown;
    if (typeof v === "string") { try { v = JSON.parse(v); } catch (_) { return v; } }
    if (v && typeof v === "object" && !Object.keys(v).length) return "";
    return JSON.stringify(v, null, 2);
  }

  function namingRow(r) {
    const row = r || {};
    const refused = asList(row.refused_reasons);
    return {
      id: String(row.naming_id || ""),
      at: fmtAt(row.written_at),
      verdict: String(row.verdict || "?"),
      name: row.name ? String(row.name) : "—",
      sentence: row.sentence ? String(row.sentence) : "",
      merge_with: row.merge_with ? String(row.merge_with) : "",
      refused: refused.length ? refused.join(", ") : "",
      evidence_refs: asList(row.evidence_refs).join(", "),
      packet: packetText(row.shown),
      packet_id: String(row.packet_id || ""),
    };
  }

  // Echo: evidence rows whose meta says they quote the node's own current name. Counted only over
  // the rows loaded; rows from before durable-runs stamped the mark carry no key, and that is said.
  function metaOf(row) {
    const m = row && row.meta;
    if (m && typeof m === "object") return m;
    if (typeof m === "string") { try { const p = JSON.parse(m); return p && typeof p === "object" ? p : {}; } catch (_) { return {}; } }
    return {};
  }

  function echoSummary(rows) {
    const list = Array.isArray(rows) ? rows : [];
    const metas = list.map(metaOf);
    const marked = metas.filter((m) => Object.prototype.hasOwnProperty.call(m, "cites_own_name")).length;
    const cites = metas.filter((m) => m.cites_own_name === true).length;
    if (!list.length) return { cites: 0, marked: 0, n: 0, text: "" };
    if (!marked) {
      return { cites: 0, marked: 0, n: list.length,
               text: `Echo: none of the ${list.length} loaded evidence rows carry the cites_own_name mark yet.` };
    }
    return { cites, marked, n: list.length,
             text: `Echo: ${cites} of ${list.length} loaded evidence rows quote this node's own name` +
               (marked < list.length ? ` (${list.length - marked} ${list.length - marked === 1 ? "row predates" : "rows predate"} the mark).` : ".") };
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
    // window_days is the span the frame really covers (1..14), not a fixed 14.
    const span = Number(s.window_days) === 1 ? "day" : `${s.window_days} days`;
    return `In curiosity over the last ${span}: took the top suggestion ${s.took_top} of ` +
      `${s.choices} times, went their own way ${s.went_own_way} of ${s.choices}.`;
  }

  function skipText(cue) {
    if (!cue || !cue.skip_reason) return "";
    return `${cue.skip_reason}: ${SKIP_TEXT[cue.skip_reason] || "no cue rendered."}`;
  }

  const api = {
    escapeHtml, fmtAt, fmtNum, fmtSigned, lanesText, sparklineModel, sparklineSvg, nodesFrom, evidenceFrom,
    situationText, situationDetail, NO_SITUATION_TEXT, nameView, namingFrom, namingRow, packetText, echoSummary, dispositionRow, evidenceRow, pageState, countsText, shownCountText, renderParamsText, choiceStyleText, skipText, EMPTY_TEXT, NODE_LIMIT,
    EVIDENCE_LIMIT,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // --- DOM layer ------------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const view = { seq: 0, frame: null, nodes: new Map() };
  const COLUMNS = ["Label", "Name", "Weight", "Evidence", "Lanes", "When", "Chosen", "First seen", "Last reinforced"];
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

  function renderNaming(box, rows, err) {
    box.replaceChildren(el("h4", null, "Naming history"));
    if (err) {
      box.append(el("div", { class: err.status === 404 ? "muted" : "err" }, err.status === 404
        ? `No naming log from durable-runs (${err.message}).` : `Could not load naming history: ${err.message}`));
      return;
    }
    if (!rows.length) { box.append(el("div", { class: "muted" }, "Orion has not been shown this node to name yet.")); return; }
    const table = el("table", { class: "evidence naming" });
    const head = el("tr");
    ["When", "Verdict", "Name", "Sentence", "Refused because", "Evidence cited", "Packet shown"]
      .forEach((h) => head.append(el("th", null, h)));
    table.append(head);
    rows.map(namingRow).forEach((r) => {
      const tr = el("tr", { title: r.id });
      tr.append(el("td", { class: "time" }, r.at));
      tr.append(el("td", null, r.merge_with ? `${r.verdict} → ${r.merge_with}` : r.verdict));
      tr.append(el("td", null, r.name));
      tr.append(el("td", { class: "excerpt" }, r.sentence));
      tr.append(el("td", { class: r.refused ? "err" : "" }, r.refused));
      tr.append(el("td", null, r.evidence_refs));
      const packet = el("td");
      if (r.packet) {
        const det = el("details");
        det.append(el("summary", null, r.packet_id || "packet"), el("pre", { class: "packet" }, r.packet));
        packet.append(det);
      } else packet.textContent = "—";
      tr.append(packet);
      table.append(tr);
    });
    box.append(table);
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
    const nameBox = el("div", { class: "naming-box" });
    if (row.naming.sentence) nameBox.append(el("div", { class: "name-sentence" }, `Orion: "${row.naming.sentence}"`));
    const historyBox = el("div", { class: "naming-history" }, "loading naming history…");
    const echoBox = el("div", { class: "muted echo" });
    const evBox = el("div", null);
    evBox.append(el("div", { class: "muted" }, "loading evidence…"));
    cell.append(nameBox, historyBox, echoBox, evBox);
    holder.append(cell);
    tr.after(holder);
    const id = encodeURIComponent(row.id);
    const naming = getJson(`/api/personality/naming?disposition_id=${id}`)
      .then((body) => { if (holder.isConnected) renderNaming(historyBox, namingFrom(body), null); })
      .catch((err) => { if (holder.isConnected) renderNaming(historyBox, [], err); });
    try {
      const body = await getJson(`/api/personality/evidence?disposition_id=${id}&limit=${EVIDENCE_LIMIT}`);
      if (!holder.isConnected) return;   // closed (or refreshed away) while loading
      const rows = evidenceFrom(body);
      echoBox.textContent = echoSummary(rows).text;
      renderEvidence(evBox, rows);
    } catch (err) {
      if (holder.isConnected) evBox.replaceChildren(el("div", { class: "err" }, `Could not load evidence: ${err.message}`));
    }
    await naming;
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
    const lookup = (id) => {
      const t = view.nodes.get(id);
      return t ? (t.name || t.label || id) : null;
    };
    nodes.map((n) => ({ ...dispositionRow(n), naming: nameView(n, lookup) })).forEach((r, i) => {
      const tr = el("tr", { class: "node", title: `${r.origin} · ${r.id} — click for evidence` });
      // The label is a real button (keyboard + screen reader); the whole row is also clickable.
      const button = el("button", { type: "button", class: "label-btn", "aria-expanded": "false",
                                    "aria-controls": `${boxId}-ev-${i}` }, r.label);
      const labelCell = el("td", { class: "label" });
      labelCell.append(button);
      tr.append(labelCell);
      const nameCell = el("td", { class: "name" });
      if (r.naming.name) nameCell.append(el("span", { class: "node-name", title: r.naming.sentence || "" }, r.naming.name), " ");
      nameCell.append(el("span", { class: `badge naming-${r.naming.badge.kind}` }, r.naming.badge.text));
      if (r.naming.grounding) nameCell.append(el("div", { class: "muted grounding" }, r.naming.grounding));
      tr.append(nameCell);
      const w = el("td", { class: "num weight" });
      const spark = el("span", { class: "spark-wrap" });
      spark.innerHTML = sparklineSvg(r.history);   // numbers + escaped title only
      w.append(spark, " ", el("span", null, r.weight));
      tr.append(w);
      tr.append(el("td", { class: "num" }, r.evidence));
      tr.append(el("td", { class: "lanes" }, r.lanes));
      tr.append(el("td", { class: "when", title: r.when_detail }, r.when));   // textContent: escaped
      tr.append(el("td", { class: "num" }, r.chosen));
      tr.append(el("td", { class: "time" }, r.first_seen));
      tr.append(el("td", { class: "time" }, r.last_reinforced));
      tr.addEventListener("click", () => toggleEvidence(tr, r, button));
      table.append(tr);
    });
    scroll.append(table);
    box.append(scroll);
  }

  const NODE_TARGETS = { active: ["pActive", "pActiveCount", "No active nodes yet."],
                         candidate: ["pCandidates", "pCandidateCount", "No candidates."],
                         retired: ["pRetired", "pRetiredCount", "Nothing retired."] };

  async function fetchNodes(status) {
    try {
      return { status, nodes: nodesFrom(await getJson(`/api/personality/dispositions?status=${status}&limit=${NODE_LIMIT}`)) };
    } catch (err) {
      return { status, err };
    }
  }

  // All three groups land before any renders, so a "merged into X" badge resolves X the same way
  // on every load, against this refresh's nodes only.
  async function loadNodes(seq) {
    const results = await Promise.all(["active", "candidate", "retired"].map(fetchNodes));
    if (seq !== view.seq) return;
    view.nodes = new Map();
    results.forEach((r) => (r.nodes || []).forEach((n) => {
      if (n && n.disposition_id) view.nodes.set(String(n.disposition_id), n);
    }));
    results.forEach(({ status, nodes, err }) => {
      const target = NODE_TARGETS[status];
      if (err) {
        $(target[1]).textContent = "";
        $(target[0]).replaceChildren(el("div", { class: "err" }, `Could not load ${status} nodes: ${err.message}`));
        return;
      }
      renderNodes(target[0], target[1], nodes, target[2], (view.frame || {})[`${status}_count`]);
    });
  }


  function clearNodes() {
    view.nodes.clear();
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
    await loadNodes(seq);
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

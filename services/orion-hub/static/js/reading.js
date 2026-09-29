/* Reading operator panel (templates/reading.html, scripts/world_pulse_read_routes.py).
 *
 * Pure functions (statusLabel, sourceLabel, allowedActions, ...) are exported for
 * node:test. Read content is web-derived, so the DOM layer only ever sets
 * textContent and never parses data as markup. Endpoints:
 *   GET  /world-pulse-read/api/status               wallets + queue counts
 *   GET  /world-pulse-read/api/reads?phase=&kind=   newest-first list
 *   GET  /world-pulse-read/api/reads/{seed_id}      full outputs
 *   POST /world-pulse-read/api/reads                submit a URL
 *   POST /world-pulse-read/api/reads/{id}/cancel    cancel
 *   POST /world-pulse-read/api/reads/{id}/retry     retry {stage}
 */
(function (root) {
  "use strict";

  const BASE = "/world-pulse-read/api";
  const PAGE_SIZE = 50;
  const POLL_MS = 15000;

  const STATUS = {
    queued: ["waiting to be read", "busy"],
    started: ["reading now (stage 1)", "busy"],
    stage1_completed: ["read; follow-up waiting (stage 2)", "busy"],
    stage2_started: ["follow-up running (stage 2)", "busy"],
    landing_pending: ["done; saving to journal", "busy"],
    completed: ["done", "good"],
    failed: ["failed", "bad"],
    skipped: ["skipped", "warn"],
  };

  function statusLabel(status) {
    const hit = STATUS[status];
    return hit ? { text: hit[0], tone: hit[1] } : { text: String(status || "unknown"), tone: "warn" };
  }

  // A duplicate request is folded into its target read; its own row is never read.
  // already_read: the URL was read before, so this row is passed on by design.
  function rowStatus(item) {
    if (item && (item.last_error === "already_read" || item.stage2_error === "already_read")) {
      return { text: "already read, passed on", tone: "warn" };
    }
    if (item && item.duplicate_of) return { text: "merged into another read", tone: "warn" };
    return statusLabel(item && item.reading_status);
  }

  function submitText(out) {
    if (out && out.duplicate === "already_read") {
      return "Not queued: Orion already read this URL, so it is blocked as a duplicate by design. Showing the earlier read.";
    }
    if (out && out.duplicate === "already_queued") return "Already queued: joined the read that is in progress.";
    const pos = out && out.queue_position ? ` — ${out.queue_position} of ${out.queue_depth} in line` : "";
    return `Queued: ${statusLabel(out && out.status).text}${pos}`;
  }

  function actionResultText(label, out) {
    if (out && out.run_already_finished) {
      return `${label}: too late, the run had already ${out.durable_status}; its result will be recorded.`;
    }
    const extra = out && out.durable_status ? ` (run is now ${out.durable_status})` : "";
    return `${label}: ${(out && out.action) || "done"}${extra}`;
  }

  function sourceLabel(item) {
    const by = (item && item.requested_by) || "world_pulse";
    const ctx = (item && (item.invocation_context || (item.request && item.request.invocation_context))) || "";
    if (by === "juniper") return ctx === "operator" ? "Juniper (Hub)" : "Juniper (chat)";
    if (by === "orion") return "Orion (curiosity)";
    return "World Pulse";
  }

  function hasEvidence(detail) {
    const ev = detail && detail.handoff && detail.handoff.read_evidence;
    return Array.isArray(ev) && ev.length > 0;
  }

  function unconsumedStage(detail, stage) {
    return ((detail && detail.durable_turns) || []).some((t) => t.stage === stage && !t.consumed_at);
  }

  // A hint only; the server re-checks every rule and returns a refusal code.
  function allowedActions(detail) {
    if (!detail || detail.duplicate_of) return { cancel: false, retry1: false, retry2: false };
    const s1 = detail.status, s2 = detail.stage2_status;
    const active = s1 === "pending" || s1 === "claimed" || (s1 === "done" && (s2 === "pending" || s2 === "claimed"));
    return {
      cancel: active,
      retry1: (s1 === "failed" || s1 === "skipped") && !["stale_digest_item", "already_read"].includes(detail.last_error) && !unconsumedStage(detail, 1),
      retry2: s1 === "done" && (s2 === "failed" || s2 === "skipped") && detail.stage2_error !== "already_read" && hasEvidence(detail) && !unconsumedStage(detail, 2),
    };
  }

  const REFUSALS = {
    not_active: "Nothing to cancel: this read is not waiting or running.",
    claimed_without_binding_retry_shortly: "A worker just picked this up; try again in a few seconds.",
    alias_row_act_on_target: "This request is a duplicate of another read; act on that one instead.",
    active_durable_binding: "A run for this stage is still open; cancel it first or wait for it to finish.",
    stage1_not_terminal: "Stage 1 hasn't failed or been skipped, so there is nothing to retry.",
    stage2_not_terminal: "Stage 2 hasn't failed or been skipped, so there is nothing to retry.",
    stage1_not_done: "Stage 2 needs a finished Stage 1 first.",
    no_read_evidence: "Stage 1 never actually fetched the source, so Stage 2 has nothing to work from. Retry Stage 1.",
    url_already_active: "Another read of this same URL is already queued or running.",
    already_read: "Orion already read this URL, so it is blocked as a duplicate by design.",
    stale_digest_item_would_be_reskipped: "Old digest items are skipped automatically, so a retry would be skipped again. Use \"Read this URL again\" to queue it as a new read.",
    durable_run_not_found_retry_shortly: "The run hasn't reached the run service yet; try again shortly.",
    reading_control_requires_hub_page: "Controls only work from the Hub page.",
    reading_db_unavailable: "Hub has no database connection right now.",
    document_reading_disabled: "Reading documents by path is turned off (HUB_READING_DOCUMENT_ROOTS is empty).",
    document_outside_allowed_roots: "That path is outside the folders Orion is allowed to read.",
    document_path_denied: "That path is blocked (secrets, keys and .git are never read).",
    document_type_not_allowed: "Only text documents can be read (e.g. .md, .txt, .rst).",
    document_not_found: "No file exists at that path (as Hub sees the disk).",
    document_not_a_file: "That path is not a regular file.",
    document_unreadable: "Hub could not open that file.",
    document_changed_during_read: "That path changed while Hub was reading it, so it was refused.",
    document_snapshot_missing: "Hub never captured that exact version from that path.",
    document_too_large: "That document is over the size limit, so it was refused rather than read in part.",
    document_not_text: "That file is not UTF-8 text.",
    document_empty: "That document is empty.",
    invalid_document_path: "That doesn't look like an absolute file path.",
  };

  // file:///abs/path?sha256=<hex> -> "/abs/path (version abcdef123456)".
  function documentLabel(url) {
    const m = /^file:\/\/(\/[^?#]*)(?:\?sha256=([0-9a-f]{64}))?$/.exec(String(url || ""));
    if (!m) return null;
    let path = m[1];
    try { path = decodeURIComponent(path); } catch (_) { /* keep raw */ }
    return m[2] ? `${path} (version ${m[2].slice(0, 12)})` : path;
  }

  function refusalText(code) {
    return REFUSALS[code] || String(code || "request failed");
  }

  function listQuery(state) {
    const p = new URLSearchParams();
    p.set("phase", state.phase || "all");
    if (state.kind) p.set("kind", state.kind);
    if (state.includeStale) p.set("include_stale", "true");
    p.set("limit", String(PAGE_SIZE));
    p.set("offset", String(state.offset || 0));
    return p.toString();
  }

  function fmtAt(iso) {
    const t = iso ? Date.parse(iso) : NaN;
    if (!isFinite(t)) return "";
    return new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function walletText(w) {
    if (!w) return "unknown";
    return `${Number(w.done_today || 0)} read today${w.enabled ? "" : " (paused)"}`;
  }

  const api = {
    statusLabel, rowStatus, submitText, actionResultText, sourceLabel, allowedActions, refusalText,
    listQuery, fmtAt, walletText, hasEvidence, documentLabel,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.OrionReading = api;
  if (typeof document === "undefined") return;

  // ---------------------------------------------------------------- DOM layer
  const $ = (id) => document.getElementById(id);
  const view = { phase: "all", kind: "", includeStale: false, offset: 0, total: 0, selected: null, selectedBusy: false };

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (v === null || v === undefined || v === false) return;
      if (k === "text") node.textContent = String(v);
      else if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : String(v));
    });
    [].concat(children || []).forEach((c) => {
      if (c === null || c === undefined || c === false) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function badge(item) {
    const s = rowStatus(item);
    return el("span", { class: `badge tone-${s.tone}`, text: s.text });
  }

  function safeLink(url, label) {
    const ok = /^https?:\/\//i.test(String(url || ""));
    return ok ? el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: label || url })
              : el("span", { text: label || documentLabel(url) || url || "" });
  }

  async function getJson(path) {
    const res = await fetch(path, { credentials: "same-origin", cache: "no-store" });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(refusalText(body.detail));
    return body;
  }

  async function postJson(path, body) {
    const res = await fetch(path, {
      method: "POST", credentials: "same-origin",
      // X-Requested-With: the server refuses controls without it (cross-site request forgery guard).
      headers: { "Content-Type": "application/json", "X-Requested-With": "orion-hub" },
      body: JSON.stringify(body || {}),
    });
    const out = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(out.detail) ? "invalid request" : out.detail;
      throw new Error(refusalText(detail));
    }
    return out;
  }

  function stat(label, big, sub) {
    return el("div", { class: "stat" }, [
      el("div", { class: "muted", text: label }),
      el("div", { class: "big", text: big }),
      sub ? el("div", { class: "muted", text: sub }) : null,
    ]);
  }

  async function loadStatus() {
    const box = $("stats");
    try {
      const s = await getJson(`${BASE}/status`);
      const q = s.queue || {}, q2 = s.stage2_queue || {}, r = s.retries || {};
      box.replaceChildren(
        stat("First reads (wallet A)", walletText(s.wallet_a),
             s.wallet_a && s.wallet_a.retry_not_before ? `backing off until ${fmtAt(s.wallet_a.retry_not_before)}` : `last ${fmtAt(s.last_stage1_at) || "never"}`),
        stat("Follow-ups (wallet B)", walletText(s.wallet_b),
             s.wallet_b && s.wallet_b.retry_not_before ? `backing off until ${fmtAt(s.wallet_b.retry_not_before)}` : `last ${fmtAt(s.last_stage2_at) || "never"}`),
        stat("Stage 1 queue", `${q.pending || 0} waiting · ${q.claimed || 0} running`,
             `${q.done || 0} done · ${q.failed || 0} failed · ${q.skipped || 0} skipped`),
        stat("Stage 2 queue", `${q2.pending || 0} waiting · ${q2.claimed || 0} running`,
             `${q2.done || 0} done · ${q2.failed || 0} failed · ${q2.skipped || 0} skipped`),
        stat("Retries", `${(r.stage1_pending_retry || 0) + (r.stage2_pending_retry || 0)} waiting to retry`,
             `${(r.stage1_exhausted || 0) + (r.stage2_exhausted || 0)} out of attempts (max ${r.max_attempts || "?"})`),
      );
      if (!s.available) box.appendChild(el("div", { class: "muted", text: "Status partly unavailable (no Redis or database)." }));
    } catch (err) {
      box.replaceChildren(el("span", { class: "muted", text: `Status unavailable: ${err.message}` }));
    }
  }

  async function loadList() {
    const body = $("reads");
    let data;
    try {
      data = await getJson(`${BASE}/reads?${listQuery(view)}`);
    } catch (err) {
      body.replaceChildren(el("tr", {}, el("td", { colspan: 4, class: "muted", text: `Could not load reads: ${err.message}` })));
      return;
    }
    view.total = data.total || 0;
    $("listTotal").textContent = `(${view.total})`;
    const items = data.items || [];
    if (!items.length) {
      body.replaceChildren(el("tr", {}, el("td", { colspan: 4, class: "muted", text: "No reads match this filter." })));
    } else {
      body.replaceChildren(...items.map((item) => el("tr", {
        class: `clickable${item.seed_id === view.selected ? " selected" : ""}`,
        "data-seed": item.seed_id,
        onclick: () => select(item.seed_id),
      }, [
        el("td", { class: "title-cell" }, [
          el("div", { class: "t", text: item.title || documentLabel(item.url) || item.url }),
          item.title ? el("div", { class: "u", text: documentLabel(item.url) || item.url }) : null,
          item.preview ? el("div", { class: "preview", text: item.preview }) : null,
        ]),
        el("td", { text: sourceLabel(item) }),
        el("td", {}, [badge(item),
                      (item.stage2_error || item.last_error) ? el("div", { class: "muted", text: item.stage2_error || item.last_error }) : null]),
        el("td", { class: "muted", text: fmtAt(item.updated_at) }),
      ])));
    }
    const end = Math.min(view.offset + PAGE_SIZE, view.total);
    $("pageInfo").textContent = view.total ? `${view.offset + 1}–${end} of ${view.total}` : "";
    $("prevPage").disabled = view.offset <= 0;
    $("nextPage").disabled = end >= view.total;
    $("refreshedAt").textContent = `updated ${fmtAt(new Date().toISOString())}`;
  }

  function list(items, render) {
    if (!Array.isArray(items) || !items.length) return el("div", { class: "muted", text: "none" });
    return el("ul", { class: "items" }, items.map((x) => el("li", {}, render(x))));
  }

  function priorsList(priors) {
    return list(priors, (p) => [
      el("span", { class: "badge", text: `${Math.round(Number(p.confidence || 0) * 100)}% sure` }), " ",
      el("span", { text: p.claim || "" }),
    ]);
  }

  function conceptsList(concepts) {
    return list(concepts, (c) => [
      el("strong", { text: c.label || "" }),
      c.definition ? el("span", { text: ` — ${c.definition}` }) : null,
      Array.isArray(c.link_hints) && c.link_hints.length ? el("div", { class: "muted", text: `links to: ${c.link_hints.join(", ")}` }) : null,
    ]);
  }

  function stage1Block(d) {
    const h = d.handoff;
    if (!h) return el("div", { class: "muted", text: "No Stage 1 output saved for this read." });
    const box = el("div", { class: d.handoff_accepted ? "" : "rejected" }, [
      d.handoff_accepted ? null : el("div", { class: "note",
        text: "Rejected, not learned: Stage 1 did not finish for this read, so this write-up was not accepted." }),
      el("h3", { text: "What Orion took from the source" }),
      el("div", { class: "prose", text: h.what_i_learned || "" }),
      el("h3", { text: "Beliefs they might adopt" }), priorsList(h.candidate_priors),
      el("h3", { text: "New concepts" }), conceptsList(h.concept_candidates),
      el("h3", { text: "Open questions" }), list(h.open_threads, (t) => t),
      el("h3", { text: "Proof they read the source" }),
      list(h.read_evidence, (e) => [el("span", { class: "badge",
        text: e.tool_name === "orion_document_snapshot" ? "Hub document snapshot" : (e.tool_name || "fetch") }), " ",
        safeLink(e.url), el("span", { class: "muted", text: ` · ${e.content_chars || 0} characters` })]),
    ]);
    return box;
  }

  function stage2Block(d) {
    const r = d.stage2_result;
    if (!r) {
      return el("div", { class: "muted", text: d.stage2_status === "done"
        ? "Stage 2 finished before results were saved on the read; see the journal entries below."
        : "No Stage 2 output yet." });
    }
    return el("div", {}, [
      el("h3", { text: "Summary" }), el("div", { class: "prose", text: r.summary || "" }),
      el("h3", { text: "Beliefs tested" }),
      list(r.priors_tested, (t) => [
        el("span", { class: `badge tone-${t.verdict === "supported" ? "good" : t.verdict === "refuted" ? "bad" : "warn"}`, text: t.verdict || "untested" }), " ",
        el("span", { text: t.claim_ref || "" }),
        t.why ? el("div", { class: "muted", text: t.why }) : null,
      ]),
      el("h3", { text: "Beliefs after the follow-up" }), priorsList(r.candidate_priors),
      el("h3", { text: "New concepts" }), conceptsList(r.concept_candidates),
      el("h3", { text: "Still open" }), list(r.open_threads, (t) => t),
      el("h3", { text: `What the follow-up did (${r.round_trips || 0} round trips)` }), list(r.hops, (t) => t),
    ]);
  }

  function actionButtons(d) {
    const can = allowedActions(d);
    const act = async (label, path, body) => {
      if (!confirm(`${label}?`)) return;
      $("actionStatus").textContent = `${label}…`;
      try {
        $("actionStatus").textContent = actionResultText(label, await postJson(path, body));
      } catch (err) {
        $("actionStatus").textContent = `${label} refused: ${err.message}`;
      }
      await Promise.all([select(d.seed_id, true), loadList(), loadStatus()]);
    };
    const id = encodeURIComponent(d.seed_id);
    return el("div", { class: "row" }, [
      el("button", { type: "button", disabled: !can.cancel, id: "cancelRead",
        title: "Stop this read. Never charges a wallet.",
        onclick: () => act("Cancel read", `${BASE}/reads/${id}/cancel`) }, "Cancel"),
      el("button", { type: "button", disabled: !can.retry1, id: "retryStage1",
        title: "Queue Stage 1 again with a fresh attempt budget.",
        onclick: () => act("Retry stage 1", `${BASE}/reads/${id}/retry`, { stage: 1 }) }, "Retry stage 1"),
      el("button", { type: "button", disabled: !can.retry2, id: "retryStage2",
        title: "Queue the follow-up again.",
        onclick: () => act("Retry stage 2", `${BASE}/reads/${id}/retry`, { stage: 2 }) }, "Retry stage 2"),
      // Escape hatch when no retry applies (e.g. an old digest item): a fresh request.
      can.cancel || d.duplicate_of || d.status === "done" || d.last_error === "already_read" ? null : el("button", { type: "button", id: "readAgain",
        title: "Put this URL in the submit form above as a new read.",
        onclick: () => {
          $("submitUrl").value = d.url || "";
          $("submitReadTitle").value = d.title || "";
          $("submitUrl").focus();
          $("submitStatus").textContent = "Add a reason if you like, then press Queue read.";
        } }, "Read this URL again"),
    ]);
  }

  function renderDetail(d) {
    const req = d.request || {};
    const meta = el("dl", { class: "meta" }, [
      el("dt", { text: "Asked by" }), el("dd", { text: sourceLabel(d) }),
      req.why_now ? el("dt", { text: "Why now" }) : null, req.why_now ? el("dd", { class: "prose", text: req.why_now }) : null,
      el("dt", { text: "Stage 1" }), el("dd", { text: `${d.status} · ${d.attempts} failed attempts${d.last_error ? ` · ${d.last_error}` : ""}` }),
      el("dt", { text: "Stage 2" }), el("dd", { text: `${d.stage2_status} · ${d.stage2_attempts} failed attempts${d.stage2_error ? ` · ${d.stage2_error}` : ""}` }),
      el("dt", { text: "Queued" }), el("dd", { text: fmtAt(d.created_at) }),
      d.completed_at ? el("dt", { text: "Stage 1 ended" }) : null, d.completed_at ? el("dd", { text: fmtAt(d.completed_at) }) : null,
      d.stage2_completed_at ? el("dt", { text: "Stage 2 ended" }) : null, d.stage2_completed_at ? el("dd", { text: fmtAt(d.stage2_completed_at) }) : null,
      d.landing_at ? el("dt", { text: "Saved to journal" }) : null, d.landing_at ? el("dd", { text: fmtAt(d.landing_at) }) : null,
      d.duplicate_of ? el("dt", { text: "Duplicate of" }) : null,
      d.duplicate_of ? el("dd", {}, el("a", { href: "#", text: d.duplicate_of, onclick: (e) => { e.preventDefault(); select(d.duplicate_of); } })) : null,
    ]);
    $("detail").replaceChildren(
      el("div", { class: "row" }, [badge(d), el("strong", { text: d.title || "" })]),
      el("div", {}, safeLink(d.url)),
      meta,
      actionButtons(d),
      el("h2", { style: "margin-top:16px", text: "Stage 1 — the read" }), stage1Block(d),
      el("h2", { style: "margin-top:16px", text: "Stage 2 — the follow-up" }), stage2Block(d),
      el("h2", { style: "margin-top:16px", text: "Journal entries" }),
      list(d.journal, (j) => el("details", {}, [
        el("summary", { text: `${j.title || j.source_ref} · ${fmtAt(j.created_at)}` }),
        el("pre", { text: j.body || "" }),
      ])),
      el("details", { style: "margin-top:12px" }, [
        el("summary", { text: "Runs, duplicates and raw data" }),
        el("h3", { text: "Runs" }),
        list(d.durable_turns, (t) => `stage ${t.stage} attempt ${t.attempt}: ${t.run_id} · ${t.consumed_at ? `collected ${fmtAt(t.consumed_at)}` : "not yet collected by the worker"}`),
        el("h3", { text: "Duplicate requests folded into this read" }),
        list(d.aliases, (a) => `${sourceLabel(a)} · ${fmtAt(a.created_at)}${a.why_now ? ` · ${a.why_now}` : ""}`),
        el("dl", { class: "meta" }, [
          el("dt", { text: "Read id" }), el("dd", { text: d.seed_id }),
          el("dt", { text: "Stage 1 trace" }), el("dd", { text: d.trace_id || "—" }),
          el("dt", { text: "Stage 2 trace" }), el("dd", { text: d.stage2_trace_id || "—" }),
        ]),
        el("pre", { text: JSON.stringify({ request: d.request, handoff: d.handoff, stage2_result: d.stage2_result }, null, 2) }),
      ]),
    );
  }

  async function select(seedId, keepStatus) {
    view.selected = seedId;
    if (!keepStatus) $("actionStatus").textContent = "";
    document.querySelectorAll("#reads tr[data-seed]").forEach((tr) => {
      tr.classList.toggle("selected", tr.getAttribute("data-seed") === seedId);
    });
    try {
      const d = await getJson(`${BASE}/reads/${encodeURIComponent(seedId)}`);
      view.selectedBusy = statusLabel(d.reading_status).tone === "busy";
      renderDetail(d);
    } catch (err) {
      $("detail").replaceChildren(el("span", { class: "muted", text: `Could not load this read: ${err.message}` }));
    }
  }

  async function submit(event) {
    event.preventDefault();
    const url = $("submitUrl").value.trim();
    if (!url) return;
    $("submitButton").disabled = true;
    $("submitStatus").textContent = "Queueing…";
    try {
      const out = await postJson(`${BASE}/reads`, {
        url, why_now: $("submitWhy").value.trim(), title: $("submitReadTitle").value.trim(),
      });
      $("submitStatus").textContent = submitText(out);
      $("submitForm").reset();
      view.offset = 0;
      await loadList();
      const target = out.duplicate_of || out.seed_id;
      if (target) await select(target);
    } catch (err) {
      $("submitStatus").textContent = `Not queued: ${err.message}`;
    } finally {
      $("submitButton").disabled = false;
    }
  }

  function refreshAll(auto) {
    loadStatus();
    loadList();
    // Re-rendering collapses open sections; only poll a detail that can still change.
    if (view.selected && (!auto || view.selectedBusy)) select(view.selected, true);
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("phase").addEventListener("change", (e) => { view.phase = e.target.value; view.offset = 0; loadList(); });
    $("kind").addEventListener("change", (e) => { view.kind = e.target.value; view.offset = 0; loadList(); });
    $("includeStale").addEventListener("change", (e) => { view.includeStale = e.target.checked; view.offset = 0; loadList(); });
    $("refresh").addEventListener("click", () => refreshAll(false));
    $("prevPage").addEventListener("click", () => { view.offset = Math.max(0, view.offset - PAGE_SIZE); loadList(); });
    $("nextPage").addEventListener("click", () => { view.offset += PAGE_SIZE; loadList(); });
    $("submitForm").addEventListener("submit", submit);
    refreshAll(false);
    setInterval(() => { if (!document.hidden) refreshAll(true); }, POLL_MS);
  });
})(typeof window !== "undefined" ? window : globalThis);

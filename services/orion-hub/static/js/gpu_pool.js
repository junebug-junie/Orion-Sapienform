/* GPU pool operator panel (templates/gpu_pool.html, scripts/gpu_pool_routes.py).
 *
 * Pure model functions (cardModel, liveByRole, walkerPath, seriesModel, ...) are exported for
 * node:test; the DOM layer below only renders what they return. Data comes from:
 *   GET  /api/gpu-pool/state?config=1        parsed YAML + discovered roles + live leases
 *   GET  /api/gpu-pool/stream (SSE)          live state + lease events off the bus
 *   GET  /api/gpu-pool/history?minutes=N     gpu_pool_events aggregates (historical)
 *   GET  /api/gpu-pool/state?history_for=ID  one lease's path through the lease graph
 *   POST /api/gpu-pool/control               operator verbs
 */
(function (root) {
  "use strict";

  // Mirror of orion/gpu_pool/lease_graph.py _TABLE (from -> to); the walker draws these.
  const EDGES = [
    ["queued", "granted"], ["queued", "backlogged"], ["queued", "unavailable"], ["queued", "released"],
    ["backlogged", "queued"], ["backlogged", "dead_letter"], ["backlogged", "released"],
    ["granted", "released"], ["granted", "retry_wait"], ["granted", "recalling"], ["granted", "dead_letter"],
    ["recalling", "released"], ["recalling", "retry_wait"], ["recalling", "dead_letter"],
    ["retry_wait", "queued"], ["retry_wait", "unavailable"], ["retry_wait", "released"],
    ["dead_letter", "queued"], ["dead_letter", "released"],
    ["unavailable", "queued"], ["unavailable", "released"],
  ];
  const NODES = {
    queued: [90, 50], granted: [300, 50], recalling: [510, 50], released: [720, 50],
    backlogged: [90, 170], retry_wait: [300, 170], dead_letter: [510, 170], unavailable: [720, 170],
  };
  const LIVE_WINDOW = 500;

  function pct(values, q) {
    const v = values.filter((x) => typeof x === "number" && isFinite(x)).sort((a, b) => a - b);
    if (!v.length) return null;
    const idx = (v.length - 1) * q;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
  }

  function fmtAt(iso) {
    const t = iso ? Date.parse(iso) : NaN;
    return isFinite(t) ? new Date(t).toISOString().replace("T", " ").slice(0, 23) : "";
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined || !isFinite(ms)) return "–";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
    return `${(ms / 60000).toFixed(1)} min`;
  }

  const ACTIVE = (l) => l.status === "granted" || l.status === "recalling";

  /** Slots in use per role. A durable-run hold and the call running in its slot (a child,
   *  hold_lease_id set) are ONE slot: the hold counts only while none of its calls is in flight. */
  function slotUse(leases) {
    const busy = {};
    const withChild = new Set();
    (leases || []).forEach((l) => { if (ACTIVE(l) && l.hold_lease_id) withChild.add(l.hold_lease_id); });
    (leases || []).forEach((l) => {
      if (!ACTIVE(l) || !l.role) return;
      if (l.kind === "hold" && withChild.has(l.lease_id)) return;
      busy[l.role] = (busy[l.role] || 0) + 1;
    });
    return busy;
  }

  /** Per card: the swap state machine (idle | loading | unloading | fault), the action in flight
   *  or last finished, and whether the pool actuates a seat there or only reports. */
  function swapModel(state) {
    const out = {};
    (state && state.cards || []).forEach((c) => {
      out[c.card] = {
        swapState: c.swap_state || "idle", swapRole: c.swap_role || null, actuatedRoles: c.actuated_roles || [],
        action: c.actuation || null, cooldownUntil: c.cooldown_until || null,
        residencyUntil: c.residency_until || null, loadedAt: c.loaded_at || null,
      };
    });
    return out;
  }

  /** Swap-load guards as the pool last read them: [{name, clear, why}]. */
  function guardModel(state) {
    return Object.entries((state && state.swap_guards) || {}).map(([name, why]) => ({ name, clear: why === null, why }));
  }

  /** Durable-run holds, each with the calls made under it (children), newest first. */
  function holdModel(state) {
    const leases = (state && state.leases) || [];
    const children = {};
    leases.forEach((l) => { if (l.hold_lease_id) (children[l.hold_lease_id] = children[l.hold_lease_id] || []).push(l); });
    return leases.filter((l) => l.kind === "hold").map((h) => {
      const kids = children[h.lease_id] || [];
      return { leaseId: h.lease_id, holder: h.holder, workClass: h.work_class, priority: h.priority, status: h.status,
               role: h.role || null, generation: h.generation || 0, grantedAt: h.granted_at || null,
               recallBy: h.recall_by || null, children: kids, inFlight: kids.filter(ACTIVE).length,
               waiting: kids.filter((k) => k.status === "queued").length };
    }).sort((a, b) => String(b.grantedAt || "").localeCompare(String(a.grantedAt || "")));
  }

  /** Cards with the roles that live on them, from the parsed YAML plus live discovery. */
  function cardModel(config, state) {
    if (!config) return { cards: [], spanning: [] };
    const discovered = {};
    (state && state.roles || []).forEach((r) => { discovered[r.role] = r; });
    const busy = slotUse(state && state.leases);
    const cardState = {};
    (state && state.cards || []).forEach((c) => { cardState[c.card] = c; });
    const classes = config.classes || {};
    const roles = Object.entries(config.roles || {}).map(([name, spec]) => {
      const d = discovered[name] || {};
      const owners = spec.owner || [];
      const borrowers = Object.entries(classes)
        .filter(([cls, c]) => (c.roles || []).includes(name) && !owners.includes(cls)).map(([cls]) => cls);
      return {
        name, kind: spec.kind, cards: spec.cards || [], swap: !!spec.swap, operatorOnly: !!spec.operator_only,
        owners, borrowers, status: d.status || "unknown", profile: d.profile_name || null,
        modelFile: d.model_file || null, slots: d.slots || 0, busy: busy[name] || 0,
        ctx: d.ctx_per_slot || null, detail: d.detail || null, port: spec.port,
      };
    });
    const swaps = swapModel(state);
    const cards = Object.entries(config.cards || {}).map(([card, spec]) => ({
      card, vramGb: spec.vram_gb, lendable: !!spec.lendable, lent: !!(cardState[card] && cardState[card].lent),
      swappedIn: (cardState[card] && cardState[card].swapped_in) || [],
      swap: swaps[card] || { swapState: "idle", actuatedRoles: [], action: null },
      roles: roles.filter((r) => r.cards.length === 1 && r.cards[0] === card),
    }));
    return { cards, spanning: roles.filter((r) => r.cards.length > 1) };
  }

  /** Live per-role view: slots in use now (state) + recent grant waits / failures (event buffer). */
  function liveByRole(state, events) {
    const model = {};
    (state && state.roles || []).forEach((r) => {
      model[r.role] = { role: r.role, status: r.status, slots: r.slots, busy: 0, grants: 0, failures: 0,
                        recalls: 0, waits: [] };
    });
    Object.entries(slotUse(state && state.leases)).forEach(([role, n]) => { if (model[role]) model[role].busy = n; });
    (events || []).forEach((e) => {
      const m = e.role && model[e.role];
      if (!m) return;
      if (e.event === "granted") { m.grants += 1; m.waits.push(e.waited_ms); }
      if (["aborted", "expired", "dead_lettered", "unavailable"].includes(e.event)) m.failures += 1;
      if (e.event === "recalled") m.recalls += 1;
    });
    return Object.values(model).map((m) => ({ ...m, wait_p50_ms: pct(m.waits, 0.5), wait_p95_ms: pct(m.waits, 0.95) }));
  }

  /** Live per class/holder/priority from the event buffer (same columns as the SQL view). */
  function liveByClass(events) {
    const groups = {};
    (events || []).forEach((e) => {
      if (!e.work_class) return;
      const key = [e.work_class, e.holder, e.priority].join("|");
      const g = groups[key] || (groups[key] = { work_class: e.work_class, holder: e.holder, priority: e.priority,
                                                grants: 0, failures: 0, recalls: 0, backlogged: 0, waits: [] });
      if (e.event === "granted") { g.grants += 1; g.waits.push(e.waited_ms); }
      if (["aborted", "expired", "dead_lettered", "unavailable"].includes(e.event)) g.failures += 1;
      if (e.event === "recalled") g.recalls += 1;
      if (e.event === "backlogged") g.backlogged += 1;
    });
    return Object.values(groups).map((g) => ({ ...g, wait_p50_ms: pct(g.waits, 0.5), wait_p95_ms: pct(g.waits, 0.95) }))
      .sort((a, b) => b.grants - a.grants);
  }

  /** The path a lease took: ordered statuses and the traversed edges, with time spent in each. */
  function walkerPath(history) {
    const steps = [];
    let prevAt = null;
    (history || []).forEach((h) => {
      const at = h.at ? Date.parse(h.at) : null;
      steps.push({ event: h.event, from: h.from || null, to: h.status, at: h.at, role: h.role || null,
                   reason: h.reason || null, attempt: h.attempt || null,
                   sincePrevMs: at !== null && prevAt !== null ? at - prevAt : null });
      if (at !== null) prevAt = at;
    });
    const edges = steps.filter((s) => s.from && s.to && s.from !== s.to).map((s) => [s.from, s.to]);
    const visited = new Set(steps.map((s) => s.to));
    const current = steps.length ? steps[steps.length - 1].to : null;
    return { steps, edges, visited: Array.from(visited), current };
  }

  /** Time series for the SVG: total grants per bucket and the worst p95 wait in it. */
  function seriesModel(rows) {
    const byT = {};
    (rows || []).forEach((r) => {
      const t = Date.parse(r.t);
      const b = byT[t] || (byT[t] = { t, grants: 0, p95: null });
      b.grants += Number(r.grants || 0);
      if (r.wait_p95_ms !== null && r.wait_p95_ms !== undefined) b.p95 = Math.max(b.p95 || 0, Number(r.wait_p95_ms));
    });
    return Object.values(byT).sort((a, b) => a.t - b.t);
  }

  /** Seconds since the pool last published state, or null if unknown. */
  function stateAgeSec(state, nowMs) {
    const t = state && state.generated_at ? Date.parse(state.generated_at) : NaN;
    return isFinite(t) ? Math.max(0, (nowMs - t) / 1000) : null;
  }

  /** The class an operator hold must request for a role (never assume class name == role name). */
  function holdClassFor(config, role) {
    const hit = Object.entries((config && config.classes) || {}).find(([, c]) => (c.roles || []).includes(role));
    return hit ? hit[0] : null;
  }

  const BACKFILL_LIMIT = 1000;  // the pool's own cap
  function backfillLabel(n) {
    return n >= BACKFILL_LIMIT ? `${BACKFILL_LIMIT}+` : String(n);
  }

  const api = { EDGES, NODES, pct, fmtMs, fmtAt, stateAgeSec, holdClassFor, backfillLabel, BACKFILL_LIMIT, cardModel, liveByRole, liveByClass, walkerPath, seriesModel,
                slotUse, swapModel, guardModel, holdModel };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.OrionGpuPool = api;
  if (typeof document === "undefined") return;

  // ---------------------------------------------------------------- DOM layer
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const view = { config: null, configYaml: "", state: null, events: [], range: "live", history: null,
                 streamUp: false, previewedBackfill: null, configLoading: false };
  const STALE_AFTER_SEC = 20;  // the pool publishes every ~5 s

  function renderCards() {
    const m = cardModel(view.config, view.state);
    const roleHtml = (r) => {
      const slots = Array.from({ length: Math.max(r.slots, 0) }, (_, i) => `<span class="${i < r.busy ? "busy" : ""}"></span>`).join("");
      return `<div class="role status-${esc(r.status)} ${r.swap ? "swap" : ""}" data-role="${esc(r.name)}">
        <div><span class="name">${esc(r.name)}</span> <span class="badge">${esc(r.status)}</span>
          ${r.swap ? '<span class="badge">swap seat</span>' : ""}${r.operatorOnly ? ' <span class="badge">operator only</span>' : ""}</div>
        <div class="model">${esc(r.modelFile || (r.kind === "service" ? "service" : "no model loaded"))}</div>
        <div class="meta">${r.profile ? esc(r.profile) + " · " : ""}${r.slots ? `${r.busy}/${r.slots} slots in use` : "no slots"}${r.ctx ? ` · ${r.ctx} ctx/slot` : ""} · :${esc(r.port)}</div>
        <div class="meta">owners: ${esc(r.owners.join(", ") || "–")}${r.borrowers.length ? ` · may borrow: ${esc(r.borrowers.join(", "))}` : ""}</div>
        ${r.detail ? `<div class="meta">${esc(r.detail)}</div>` : ""}
        ${r.slots ? `<div class="slotbar" aria-hidden="true">${slots}</div>` : ""}
      </div>`;
    };
    const swapHtml = (c) => {
      const sw = c.swap;
      const seats = c.roles.filter((r) => r.swap).map((r) => r.name);
      if (!seats.length && sw.swapState === "idle" && !sw.action) return "";
      const a = sw.action;
      const lines = [];
      lines.push(sw.actuatedRoles.length ? `pool actuates ${esc(sw.actuatedRoles.join(", "))}` : "observe only: swaps are reported, not actuated");
      if (a) {
        lines.push(`${esc(a.action)} ${esc(a.role)} (g${esc(a.generation)}, ${esc(a.reason)})`
          + (a.phase ? ` · phase ${esc(a.phase)}` : "") + (a.outcome ? ` · ${esc(a.outcome)}` : " · in flight")
          + (a.sent_at ? ` · sent ${esc(fmtAt(a.sent_at))}` : ""));
      }
      if (sw.cooldownUntil) lines.push(`no load before ${esc(fmtAt(sw.cooldownUntil))} (cooldown)`);
      if (sw.residencyUntil) lines.push(`residents stay until ${esc(fmtAt(sw.residencyUntil))}`);
      if (sw.loadedAt) lines.push(`seat loaded ${esc(fmtAt(sw.loadedAt))}`);
      return `<div class="swapline swap-${esc(sw.swapState)}" data-swap-state="${esc(sw.swapState)}">
        <span class="badge swapstate">swap: ${esc(sw.swapState)}${sw.swapState !== "idle" && sw.swapRole ? ` ${esc(sw.swapRole)}` : ""}</span>
        ${sw.swapState === "fault" ? '<div class="meta"><strong>FAULT</strong>: no grants on any role of this card until discovery sees it consistent again.</div>' : ""}
        ${lines.map((l) => `<div class="meta">${l}</div>`).join("")}
      </div>`;
    };
    $("cards").innerHTML = m.cards.map((c) => `<div class="card" data-card="${esc(c.card)}">
      <div class="card-head"><strong>${esc(c.card)}</strong><span class="muted">${esc(c.vramGb)} GB${c.lendable ? ` · ${c.lent ? "LENT" : "not lent"}` : ""}</span></div>
      ${swapHtml(c)}
      ${c.roles.map(roleHtml).join("") || '<div class="muted">nothing configured</div>'}
    </div>`).join("");
    const guards = guardModel(view.state);
    $("swapGuards").innerHTML = guards.length
      ? `Swap-load guards: ${guards.map((g) => `<span class="badge ${g.clear ? "guard-clear" : "guard-block"}" title="${esc(g.why || "clear")}">${esc(g.name)}: ${g.clear ? "clear" : esc(g.why)}</span>`).join(" ")}`
      : "";
    renderHolds();
    $("spanning").innerHTML = m.spanning.length
      ? `<div class="muted">Spans several cards:</div>${m.spanning.map((r) => roleHtml(r).replace('class="role', `title="${esc(r.cards.join(", "))}" class="role`)).join("")}` : "";
    const unclaimed = (view.state && view.state.unclaimed_servers) || [];
    $("unclaimed").textContent = unclaimed.length ? `Unclaimed servers (announcing, but no role in the YAML): ${unclaimed.join("; ")}` : "";
    $("yaml").textContent = view.configYaml || "";
    renderControls(m);
  }

  function renderHolds() {
    const holds = holdModel(view.state);
    if (!holds.length) {
      $("holds").innerHTML = '<div class="muted">No durable-run holds. (Durable runs move onto pool holds at the stage 4.5 cutover.)</div>';
      return;
    }
    const kid = (k) => `<tr class="clickable child" data-lease="${esc(k.lease_id)}"><td></td><td class="muted">call ${esc(k.lease_id.slice(0, 8))}</td>
      <td>${esc(k.status)}</td><td>${esc(k.role || "–")}</td><td></td><td class="muted">${esc(k.turn_correlation_id || "")}</td></tr>`;
    $("holds").innerHTML = `<div class="scroll"><table><thead><tr><th>holder</th><th>hold</th><th>status</th><th>role</th>
      <th class="num">gen</th><th>calls</th></tr></thead><tbody>${holds.map((h) => `
      <tr class="clickable hold" data-lease="${esc(h.leaseId)}"><td>${esc(h.holder)}</td><td>${esc(h.leaseId.slice(0, 8))} · ${esc(h.priority)}</td>
        <td>${esc(h.status)}${h.recallBy ? ` (give back by ${esc(fmtAt(h.recallBy))})` : ""}</td><td>${esc(h.role || "waiting")}</td>
        <td class="num">${esc(h.generation)}</td><td>${h.inFlight} running · ${h.waiting} waiting</td></tr>
      ${h.children.map(kid).join("")}`).join("")}</tbody></table></div>`;
  }

  function renderControls(m) {
    const lendable = m.cards.filter((c) => c.lendable);
    $("lendControls").innerHTML = lendable.map((c) =>
      `<button type="button" data-verb="${c.lent ? "unlend" : "lend"}" data-card="${esc(c.card)}">${c.lent ? `Take ${esc(c.card)} back (unlend)` : `Lend ${esc(c.card)}`}</button>
       <span class="muted">${c.lent ? "borrowers may use it; its owner still claws back" : "owner only"}</span>`).join("")
      || '<span class="muted">No lendable cards in the YAML.</span>';
    const classesCfg = (view.config && view.config.classes) || {};
    const holdable = m.spanning.concat(...m.cards.map((c) => c.roles)).filter((r) => r.operatorOnly);
    const holds = ((view.state && view.state.leases) || []).filter((l) => l.holder && l.holder.startsWith("operator:"));
    const enforce = view.state && view.state.mode === "enforce";
    $("holdControls").innerHTML = holdable.map((r) => {
      const cls = holdClassFor(view.config, r.name);
      const held = holds.find((l) => ((classesCfg[l.work_class] || {}).roles || []).includes(r.name));
      // An existing hold can always be released, whatever the mode.
      if (held) return `<button type="button" data-verb="release" data-lease="${esc(held.lease_id)}">Release ${esc(r.name)} (${esc(held.status)})</button>`;
      if (!enforce) return `<span class="muted">Hold ${esc(r.name)}: needs the pool to load and unload models itself (stage 5); until then a hold would only drain every card.</span>`;
      return cls ? `<button type="button" data-verb="hold" data-class="${esc(cls)}">Hold ${esc(r.name)} (drains ${esc(r.cards.join(", "))})</button>`
                 : `<span class="muted">${esc(r.name)}: no class lists this role</span>`;
    }).join("");
    const classes = Object.keys(classesCfg);
    const opts = classes.map((c) => `<option>${esc(c)}</option>`).join("");
    const sig = classes.join("|");
    if ($("bfClass").dataset.sig !== sig) { $("bfClass").innerHTML = `<option value="">any class</option>${opts}`; $("bfClass").dataset.sig = sig; }
    if ($("filterClass").dataset.sig !== sig) {
      const keep = $("filterClass").value;
      $("filterClass").innerHTML = `<option value="">all classes</option>${opts}`;
      $("filterClass").dataset.sig = sig;
      if (classes.includes(keep)) $("filterClass").value = keep;
    }
  }

  const ROLE_COLS = [["role", "role"], ["status", "status"], ["slots", "in use"], ["grants", "grants"], ["failures", "failures"],
                     ["recalls", "recalls"], ["wait_p50_ms", "wait p50"], ["wait_p95_ms", "wait p95"]];
  const CLASS_COLS = [["work_class", "class"], ["holder", "holder"], ["priority", "priority"], ["grants", "grants"],
                      ["failures", "failures"], ["recalls", "recalls"], ["backlogged", "backlogged"], ["wait_p50_ms", "wait p50"], ["wait_p95_ms", "wait p95"]];

  function table(el, cols, rows) {
    const cell = (k, r) => {
      if (k.endsWith("_ms")) return `<td class="num">${fmtMs(r[k])}</td>`;
      if (k === "slots" && "busy" in r) return `<td class="num">${r.busy}/${r.slots}</td>`;
      return typeof r[k] === "number" ? `<td class="num">${r[k]}</td>` : `<td>${esc(r[k])}</td>`;
    };
    el.innerHTML = `<thead><tr>${cols.map(([k, h]) => `<th class="${/_ms$|grants|failures|recalls|backlogged|slots/.test(k) ? "num" : ""}">${h}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${cols.map(([k]) => cell(k, r)).join("")}</tr>`).join("") || `<tr><td colspan="${cols.length}" class="muted">no traffic in this window</td></tr>`}</tbody>`;
  }

  function filtered(events) {
    const cls = $("filterClass").value, holder = $("filterHolder").value.trim();
    return events.filter((e) => (!cls || e.work_class === cls) && (!holder || (e.holder || "").includes(holder)));
  }

  function renderTraffic() {
    let byRole, byClass, events, series;
    if (view.range === "live") {
      events = filtered(view.events);
      byRole = liveByRole(view.state, events);
      byClass = liveByClass(events);
      series = seriesModel(liveSeries(events));
    } else {
      const h = view.history || { by_role: [], by_class: [], events: [], series: [] };
      byRole = h.by_role.map((r) => ({ ...r, status: "", slots: "" }));
      byClass = h.by_class;
      events = h.events;
      series = seriesModel(h.series);
    }
    table($("byRole"), ROLE_COLS, byRole);
    table($("byClass"), CLASS_COLS, byClass);
    renderSeries(series);
    $("events").innerHTML = `<thead><tr><th>time</th><th>event</th><th>class</th><th>priority</th><th>role</th><th>holder</th>
      <th class="num">waited</th><th class="num">held</th><th>attempt</th><th>reason</th><th>lease</th></tr></thead><tbody>${
      events.slice(0, 300).map((e) => `<tr class="clickable" data-lease="${esc(e.lease_id || "")}">
        <td>${esc((e.generated_at || "").replace("T", " ").slice(0, 19))}</td><td class="ev-${esc(e.event)}">${esc(e.event)}</td>
        <td>${esc(e.work_class)}</td><td>${esc(e.priority)}</td><td>${esc(e.role)}</td><td>${esc(e.holder)}</td>
        <td class="num">${fmtMs(e.waited_ms)}</td><td class="num">${fmtMs(e.held_ms)}</td><td>${esc(e.attempt)}</td>
        <td>${esc(e.reason)}</td><td class="muted">${esc((e.lease_id || "").slice(0, 8))}</td></tr>`).join("")}</tbody>`;
  }

  function liveSeries(events) {
    const buckets = {};
    events.forEach((e) => {
      if (e.event !== "granted" || !e.generated_at) return;
      const t = Math.floor(Date.parse(e.generated_at) / 60000) * 60000;
      (buckets[t] = buckets[t] || []).push(e.waited_ms);
    });
    return Object.entries(buckets).map(([t, w]) => ({ t: new Date(Number(t)).toISOString(), grants: w.length, wait_p95_ms: pct(w, 0.95) }));
  }

  function renderSeries(series) {
    const svg = $("series");
    const W = svg.clientWidth || 900, H = 160, pad = 28;
    if (!series.length) { svg.innerHTML = `<text x="${pad}" y="80">no grants in this window</text>`; return; }
    const maxG = Math.max(...series.map((s) => s.grants), 1), maxW = Math.max(...series.map((s) => s.p95 || 0), 1);
    const t0 = series[0].t, t1 = series[series.length - 1].t || t0 + 1, span = Math.max(t1 - t0, 1);
    const x = (t) => pad + ((t - t0) / span) * (W - pad * 2);
    const bw = Math.min(24, Math.max(2, (W - pad * 2) / series.length - 2));  // few buckets: bars, not slabs
    const bars = series.map((s) => `<rect x="${x(s.t) - bw / 2}" y="${H - pad - (s.grants / maxG) * (H - pad * 2)}" width="${bw}" height="${(s.grants / maxG) * (H - pad * 2)}" fill="var(--series-1)" opacity="0.7"><title>${new Date(s.t).toLocaleTimeString()}: ${s.grants} grants, p95 wait ${fmtMs(s.p95)}</title></rect>`).join("");
    const pts = series.filter((s) => s.p95 !== null).map((s) => `${x(s.t)},${H - pad - (s.p95 / maxW) * (H - pad * 2)}`).join(" ");
    svg.innerHTML = `<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="var(--axis)"/>${bars}
      ${pts ? `<polyline points="${pts}" fill="none" stroke="var(--series-2)" stroke-width="2"/>` : ""}
      <text x="${pad}" y="14">bars: grants (max ${maxG})</text><text x="${W - pad - 190}" y="14" style="fill:var(--series-2)">line: p95 wait (max ${fmtMs(maxW)})</text>
      <text x="${pad}" y="${H - 8}">${new Date(t0).toLocaleString()}</text><text x="${W - pad - 140}" y="${H - 8}">${new Date(t1).toLocaleString()}</text>`;
  }

  async function walk(leaseId) {
    if (!leaseId) return;
    $("walkLease").value = leaseId;
    const res = await fetch(`/api/gpu-pool/state?history_for=${encodeURIComponent(leaseId)}`);
    if (!res.ok) { $("walkSteps").innerHTML = `<tr><td class="muted">could not load lease: HTTP ${res.status}</td></tr>`; return; }
    const st = await res.json();
    const path = walkerPath(st.history || []);
    const onPath = new Set(path.edges.map(([a, b]) => `${a}>${b}`));
    const svg = $("walker");
    const edges = EDGES.map(([a, b]) => {
      const [x1, y1] = NODES[a], [x2, y2] = NODES[b], hit = onPath.has(`${a}>${b}`);
      if (!hit) return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--grid)" stroke-width="1"/>`;
      // stop short of the box so the arrowhead shows which way the lease moved
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, k = Math.max(0, (len - 58) / len);
      return `<line x1="${x1}" y1="${y1}" x2="${x1 + dx * k}" y2="${y1 + dy * k}" stroke="var(--series-1)" stroke-width="3" marker-end="url(#gpArrow)"/>`;
    }).join("");
    const defs = `<defs><marker id="gpArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--series-1)"/></marker></defs>`;
    const nodes = Object.entries(NODES).map(([n, [cx, cy]]) => {
      const seen = path.visited.includes(n), cur = path.current === n;
      return `<g><rect x="${cx - 55}" y="${cy - 16}" width="110" height="32" rx="8" fill="var(--surface)" stroke="${cur ? "var(--series-2)" : seen ? "var(--series-1)" : "var(--axis)"}" stroke-width="${cur ? 3 : seen ? 2 : 1}"/>
        <text x="${cx}" y="${cy + 4}" text-anchor="middle" style="fill:${seen ? "var(--ink)" : "var(--ink-muted)"}">${n}</text></g>`;
    }).join("");
    svg.setAttribute("viewBox", "0 0 810 230");
    svg.innerHTML = defs + edges + nodes;
    $("walkSteps").innerHTML = `<thead><tr><th>#</th><th>event</th><th>from → to</th><th>role</th><th>attempt</th><th class="num">after</th><th>at</th><th>reason</th></tr></thead><tbody>${
      path.steps.map((s, i) => `<tr><td>${i + 1}</td><td class="ev-${esc(s.event)}">${esc(s.event)}</td><td>${esc(s.from || "")} → ${esc(s.to)}</td>
        <td>${esc(s.role)}</td><td>${esc(s.attempt)}</td><td class="num">${fmtMs(s.sincePrevMs)}</td><td>${esc(fmtAt(s.at))}</td><td>${esc(s.reason)}</td></tr>`).join("")
      || '<tr><td colspan="8" class="muted">no history for that lease</td></tr>'}</tbody>`;
    $("walkReplay").disabled = !["dead_letter", "unavailable"].includes(path.current);
    $("walkCancel").disabled = !path.current || path.current === "released";
    $("walkReplay").dataset.lease = $("walkCancel").dataset.lease = leaseId;
  }

  async function control(body) {
    $("controlStatus").textContent = `${body.verb}…`;
    let res, out = {};
    try {
      // X-Requested-With: the server refuses control requests without it (cross-site request forgery guard).
      res = await fetch("/api/gpu-pool/control", { method: "POST", credentials: "same-origin",
                                                   headers: { "Content-Type": "application/json", "X-Requested-With": "orion-hub" },
                                                   body: JSON.stringify(body) });
      out = await res.json().catch(() => ({}));
    } catch (err) {
      $("controlStatus").textContent = `${body.verb} failed: ${err}`;
      return {};
    }
    $("controlStatus").textContent = res.ok
      ? `${body.verb}: ${out.ok ? "ok" : "refused"}${out.reason ? ` (${out.reason})` : ""}`
      : `${body.verb} failed: HTTP ${res.status} ${out.detail || ""}`;
    return out;
  }

  function backfillSpec(preview) {
    const iso = (el) => (el.value ? new Date(el.value).toISOString() : undefined);
    return { work_class: $("bfClass").value || undefined, holder: $("bfHolder").value.trim() || undefined,
             status: $("bfStatus").value || undefined, since: iso($("bfSince")), until: iso($("bfUntil")),
             limit: BACKFILL_LIMIT, preview };
  }

  function resetBackfill() {
    view.previewedBackfill = null;
    $("bfRun").disabled = true;
    $("bfRun").textContent = "Replay";
  }

  async function loadHistory() {
    if (view.range === "live") { renderTraffic(); return; }
    const q = new URLSearchParams({ minutes: view.range });
    if ($("filterClass").value) q.set("work_class", $("filterClass").value);
    if ($("filterHolder").value.trim()) q.set("holder", $("filterHolder").value.trim());
    const res = await fetch(`/api/gpu-pool/history?${q}`);
    view.history = res.ok ? await res.json() : null;
    renderTraffic();
  }

  function connect(backoff) {
    const es = new EventSource("/api/gpu-pool/stream");
    const live = (on, text) => { $("liveDot").classList.toggle("on", on); $("liveText").textContent = text; };
    const applyState = (s) => {
      if (!s) return;
      view.state = s;
      if (!view.config) { loadConfig(); return; }   // pool was unreachable at page load: try again now
      $("poolMode").textContent = `mode: ${s.mode}`;
      if (view.config && s.config_digest && view.configDigest && s.config_digest !== view.configDigest) loadConfig();
      renderCards();
      if (view.range === "live") renderTraffic();
    };
    es.addEventListener("snapshot", (e) => {
      const d = JSON.parse(e.data);
      view.events = (d.events || []).slice().reverse().slice(0, LIVE_WINDOW);
      applyState(d.state);
      view.streamUp = true;
      backoff = 2000;
      freshness();
    });
    es.addEventListener("state", (e) => { applyState(JSON.parse(e.data).state); freshness(); });
    es.addEventListener("event", (e) => {
      view.events.unshift(JSON.parse(e.data).event);
      view.events.length = Math.min(view.events.length, LIVE_WINDOW);
      if (view.range === "live") renderTraffic();
    });
    es.onerror = () => {
      view.streamUp = false;
      live(false, "disconnected, retrying");
      es.close();
      setTimeout(() => connect(Math.min((backoff || 2000) * 2, 30000)), backoff || 2000);
    };
  }

  function freshness() {
    const live = (on, text) => { $("liveDot").classList.toggle("on", on); $("liveText").textContent = text; };
    if (!view.streamUp) return;
    const age = stateAgeSec(view.state, Date.now());
    if (age === null) live(false, "connected, no pool state yet");
    else if (age > STALE_AFTER_SEC) live(false, `STALE: last pool state ${Math.round(age)} s ago -- pool may be down`);
    else live(true, `live · state ${Math.round(age)} s old`);
  }

  async function loadConfig() {
    if (view.configLoading) return;
    view.configLoading = true;
    let res;
    try {
      res = await fetch("/api/gpu-pool/state?config=1");
    } catch (err) {
      $("cardsHint").textContent = `Pool unreachable: ${err}`;
      return;
    } finally {
      view.configLoading = false;
    }
    if (!res.ok) { $("cardsHint").textContent = `Pool unreachable: HTTP ${res.status} (retrying when it publishes again)`; return; }
    const s = await res.json();
    view.config = s.config; view.configYaml = s.config_yaml; view.configDigest = s.config_digest; view.state = s;
    $("poolDigest").textContent = `config ${s.config_digest}`;
    $("poolMode").textContent = `mode: ${s.mode}`;
    renderCards();
    renderTraffic();
  }

  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-verb]");
    if (btn) {
      if (btn.disabled) return;
      const body = { verb: btn.dataset.verb };
      if (btn.dataset.card) body.card = btn.dataset.card;
      if (btn.dataset.lease) body.lease_id = btn.dataset.lease;
      if (btn.dataset.class) body.work_class = btn.dataset.class;
      if (body.verb === "hold" && !confirm(`Hold ${body.work_class}? Every card it spans is drained first.`)) return;
      btn.disabled = true;   // no double-submit before the next state frame redraws the controls
      try { await control(body); } finally { btn.disabled = false; }
      return;
    }
    const row = ev.target.closest("tr[data-lease]");
    if (row && row.dataset.lease) walk(row.dataset.lease);
  });

  document.addEventListener("DOMContentLoaded", () => {
    $("range").addEventListener("change", () => { view.range = $("range").value; loadHistory(); });
    $("refreshTraffic").addEventListener("click", loadHistory);
    $("filterClass").addEventListener("change", loadHistory);
    $("filterHolder").addEventListener("change", loadHistory);
    $("walkGo").addEventListener("click", () => walk($("walkLease").value.trim()));
    $("walkReplay").addEventListener("click", async () => { await control({ verb: "replay", lease_id: $("walkReplay").dataset.lease }); walk($("walkReplay").dataset.lease); });
    $("walkCancel").addEventListener("click", async () => { await control({ verb: "cancel", lease_id: $("walkCancel").dataset.lease }); walk($("walkCancel").dataset.lease); });
    $("bfPreview").addEventListener("click", async () => {
      const spec = backfillSpec(true);
      const out = await control({ verb: "backfill", backfill: spec });
      const n = out && out.detail ? out.detail.would_replay : undefined;
      view.previewedBackfill = n ? { ...spec, preview: false } : null;   // Run sends exactly what was previewed
      $("bfRun").disabled = !n;
      $("bfRun").textContent = n ? `Replay ${backfillLabel(n)} lease(s)` : "Replay";
    });
    ["bfClass", "bfHolder", "bfStatus", "bfSince", "bfUntil"].forEach((id) => {
      $(id).addEventListener("input", resetBackfill);
      $(id).addEventListener("change", resetBackfill);
    });
    $("bfRun").addEventListener("click", async () => {
      if (!view.previewedBackfill || !confirm($("bfRun").textContent + "?")) return;
      const spec = view.previewedBackfill;
      resetBackfill();
      await control({ verb: "backfill", backfill: spec });
    });
    setInterval(freshness, 5000);
    loadConfig().finally(() => connect(2000));
  });
})(typeof window !== "undefined" ? window : globalThis);

// Body economy tab (static/body-economy.html): how Orion's body is doing. Each limit Orion lives
// inside, what it costs now, what Orion wanted and did not get, down to single decisions.
// Reads GET /api/body-economy/report (Hub proxy of durable-runs /physiology/report,
// schema physiology.report.v1). Spec: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md
// Phase 3. An empty section always says why (report.missing, or durable-runs unreachable): never a
// blank chart.
(function () {
  "use strict";

  var WINDOWS = ["24h", "7d", "30d"];
  var HOUR_MS = 3600000;
  // Bars per window in the wants chart: one bar per bucket, buckets aligned to the window start.
  var WANT_BUCKET_HOURS = { "24h": 1, "7d": 6, "30d": 24 };

  // bind_state -> plain words. exogenous_bind is drawn as hatching, never a colour (spec: it must not
  // read as a price).
  var STATE_WORDS = {
    slack: "room to spare",
    binding: "binding",
    exogenous_bind: "over its line, not Orion's doing",
    stuck_at_ceiling: "at its ceiling",
    no_reading: "no reading",
  };
  var STATE_RANK = { stuck_at_ceiling: 0, binding: 1, exogenous_bind: 2, no_reading: 3, slack: 4 };

  // One fixed colour per want family across every view and window (dataviz reference palette, dark
  // steps). Families not listed get a stable slot from their name, never from their rank.
  var PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
  var FAMILY_SLOTS = {
    "hub.retro_leisure": 0,
    "actions.journal": 1,
    "hub.outreach": 2,
    "thought.painting": 3,
    "durable.curiosity_investigate": 4,
    "thought.dream_hop": 5,
    "durable.self_sense_eval": 6,
    "durable.journal_compose": 7,
  };
  var OTHER = "other";
  var OTHER_COLOR = "#6b7280";
  var RAMP = ["#1f2937", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec"];

  var DEFAULT_EMPTY = {
    limits: "No limit readings in this window.",
    wants: "No wants in this window.",
    rhythm: "No heavy work recorded in this window.",
    money: "No money figures yet.",
    unit_costs: "No unit costs yet.",
    waste: "No failed GPU leases in this window.",
    queue_wait: "No queued work in this window.",
    decisions: "No decisions in this window.",
    claude: "No Claude spend recorded in this window.",
    vram: "No GPU memory readings.",
    value_per_cost: "No value-per-cost figures yet.",
    services: "No service check yet.",
  };

  // ---------- small helpers ----------
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function num(v) {
    return typeof v === "number" && isFinite(v) ? v : null;
  }

  function ms(iso) {
    var t = Date.parse(iso);
    return isNaN(t) ? null : t;
  }

  function hashSlot(name) {
    var h = 0;
    for (var i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return h % PALETTE.length;
  }

  function colorForFamily(family) {
    if (family === OTHER) return OTHER_COLOR;
    var slot = Object.prototype.hasOwnProperty.call(FAMILY_SLOTS, family) ? FAMILY_SLOTS[family] : hashSlot(family);
    return PALETTE[slot];
  }

  // Local wall-clock text. tz is for tests; the browser default is the viewer's own zone.
  function fmtTime(iso, opts, tz) {
    var t = ms(iso);
    if (t === null) return "unknown";
    var o = Object.assign({ hour12: false }, opts || {});
    if (tz) o.timeZone = tz;
    return new Intl.DateTimeFormat("en-GB", o).format(new Date(t));
  }

  function fmtWhen(iso, tz) {
    return fmtTime(iso, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }, tz);
  }

  function fmtLevel(v, unit) {
    v = num(v);
    if (v === null) return "no reading";
    var u = String(unit || "");
    if (u === "C" || u === "°C") return v.toFixed(1) + " °C";
    if (u === "USD" || u === "$") return "$" + v.toFixed(2);
    if (u === "%") return Math.round(v) + "%";
    return (Math.abs(v) >= 100 ? Math.round(v) : +v.toFixed(2)) + (u ? " " + u : "");
  }

  function fmtUsd(v) {
    v = num(v);
    if (v === null) return "not measured";
    if (v !== 0 && Math.abs(v) < 0.01) return "$" + v.toPrecision(2);
    if (Math.abs(v) < 1) return "$" + v.toFixed(3);  // a kWh rate is $0.125, not $0.13
    return "$" + v.toFixed(2);
  }

  function fmtSeconds(s) {
    s = num(s);
    if (s === null) return "unknown";
    if (s < 60) return (s < 10 ? s.toFixed(1) : Math.round(s)) + " s";
    if (s < 3600) return (s / 60).toFixed(s < 600 ? 1 : 0) + " min";
    return (s / 3600).toFixed(1) + " h";
  }

  function fmtCount(n) {
    n = num(n);
    if (n === null) return "unknown";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
    return String(Math.round(n));
  }

  function statusWords(status) {
    return String(status || "unknown").replace(/_/g, " ");
  }

  function chip(text, kind) {
    return '<span class="chip chip-' + esc(kind) + '">' + esc(text) + "</span>";
  }

  function statusChip(status) {
    var s = String(status || "").replace(/ /g, "_");
    var kind = s === "measured" ? "good" : s === "derived" ? "info" : s === "estimated" ? "info" : "muted";
    return chip(statusWords(status), kind);
  }

  function stateChip(bindState) {
    var kind = bindState === "slack" ? "good" : bindState === "binding" || bindState === "stuck_at_ceiling"
      ? "warn" : "muted";
    return chip(STATE_WORDS[bindState] || statusWords(bindState), kind);
  }

  // ---------- empty states ----------
  // Why a section is empty: the API's own `missing` reason first, else the fetch failure, else a
  // plain default. Never a blank chart.
  function missingReason(report, section) {
    var list = (report && report.missing) || [];
    for (var i = 0; i < list.length; i++) {
      if (list[i] && list[i].section === section) return String(list[i].reason || "missing");
    }
    return null;
  }

  function unreachableText(err, tz) {
    if (!err) return null;
    var since = err.unreachable_since ? " since " + fmtWhen(err.unreachable_since, tz) : "";
    var what = err.status === 502 || err.unreachable_since || err.network
      ? "durable-runs unreachable" + since
      : "the report was refused (HTTP " + (err.status || "?") + ")";
    return what + (err.detail ? ": " + err.detail : "");
  }

  function emptyText(report, section, err, tz) {
    return unreachableText(err, tz) || missingReason(report, section) || DEFAULT_EMPTY[section] || "No data.";
  }

  function emptyHtml(text) {
    return '<p class="empty" data-empty="1">' + esc(text) + "</p>";
  }

  // ---------- limits ----------
  function sortedHourly(limit) {
    return ((limit && limit.hourly) || []).slice().sort(function (a, b) { return (ms(a.h) || 0) - (ms(b.h) || 0); });
  }

  // A lane is folded away when it had room to spare every hour and still does. Only GPU lanes fold:
  // heat, money and Juniper's attention always stay in view (approved mockup).
  function isIdleLimit(limit) {
    if (String((limit && limit.constraint) || "").indexOf("lane:") !== 0) return false;
    var hourly = (limit && limit.hourly) || [];
    var nowState = limit && limit.now && limit.now.bind_state;
    if (nowState && nowState !== "slack") return false;
    for (var i = 0; i < hourly.length; i++) if (hourly[i].bind_state !== "slack") return false;
    return true;
  }

  function limitRange(limit) {
    var vals = sortedHourly(limit).map(function (p) { return num(p.level); }).filter(function (v) { return v !== null; });
    var line = num(limit.line);
    if (line !== null) vals.push(line);
    if (!vals.length) return { lo: 0, hi: 1 };
    var lo = Math.min.apply(null, vals);
    var hi = Math.max.apply(null, vals);
    if (limit.unit === "%" && lo >= 0) lo = 0;
    var pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.1 || 1;
    return { lo: lo - (lo === 0 ? 0 : pad), hi: hi + pad };
  }

  // x in 0..1000 for a time inside the window.
  function xAt(t, startMs, endMs) {
    return Math.max(0, Math.min(1000, ((t - startMs) / Math.max(endMs - startMs, 1)) * 1000));
  }

  function limitSvg(limit, startMs, endMs) {
    var pts = sortedHourly(limit);
    var r = limitRange(limit);
    var H = 56;
    var y = function (v) { return (H - 2 - ((Math.min(Math.max(v, r.lo), r.hi) - r.lo) / (r.hi - r.lo)) * (H - 4)).toFixed(1); };
    var colW = (HOUR_MS / Math.max(endMs - startMs, 1)) * 1000;
    var cols = "";
    var path = "";
    var pen = "M";
    pts.forEach(function (p) {
      var t = ms(p.h);
      if (t === null) return;
      var x0 = xAt(t, startMs, endMs);
      var w = Math.max(0.5, Math.min(colW, 1000 - x0));
      var cls = { binding: "c-bind", stuck_at_ceiling: "c-ceiling", exogenous_bind: "c-exo", no_reading: "c-none" }[p.bind_state];
      if (cls) cols += '<rect class="' + cls + '" x="' + x0.toFixed(1) + '" y="0" width="' + w.toFixed(2) + '" height="' + H + '"></rect>';
      var v = num(p.level);
      if (v === null) { pen = "M"; return; }
      path += pen + (x0 + w / 2).toFixed(1) + " " + y(v) + " ";
      pen = "L";
    });
    var line = num(limit.line);
    var lineSvg = line === null ? "" : '<line class="budget" x1="0" x2="1000" y1="' + y(line) + '" y2="' + y(line) + '"></line>';
    return '<svg viewBox="0 0 1000 ' + H + '" preserveAspectRatio="none" aria-hidden="true">' +
      cols + lineSvg +
      (path ? '<path class="level" d="' + path.trim() + '"></path>' : "") + "</svg>";
  }

  // The hourly point under x (0..1) for tooltips.
  function pointAt(limit, frac, startMs, endMs) {
    var t = startMs + frac * (endMs - startMs);
    var best = null;
    sortedHourly(limit).forEach(function (p) {
      var h = ms(p.h);
      if (h !== null && h <= t && t < h + HOUR_MS) best = p;
    });
    return best;
  }

  function limitTip(limit, p, tz) {
    if (!p) return null;
    return fmtTime(p.h, { weekday: "short", hour: "2-digit", minute: "2-digit" }, tz) + " · " + fmtLevel(p.level, limit.unit) +
      " · " + (STATE_WORDS[p.bind_state] || p.bind_state) + (num(p.price) !== null ? " · price " + p.price.toFixed(2) : "");
  }

  // ---------- wants ----------
  function topFamilies(wants, n) {
    var totals = {};
    ((wants && wants.hourly) || []).forEach(function (r) {
      totals[r.family] = (totals[r.family] || 0) + (num(r.funded) || 0) + (num(r.deferred) || 0) * 1e-6;
    });
    return Object.keys(totals).sort(function (a, b) { return totals[b] - totals[a] || (a < b ? -1 : 1); }).slice(0, n || 4);
  }

  // Buckets of funded-by-family (top 4 + other) and deferred, aligned to the window start.
  function bucketWants(wants, startMs, endMs, bucketHours) {
    var size = (bucketHours || 1) * HOUR_MS;
    var n = Math.max(1, Math.ceil((endMs - startMs) / size));
    var top = topFamilies(wants, 4);
    var buckets = [];
    for (var i = 0; i < n; i++) buckets.push({ start: startMs + i * size, end: startMs + (i + 1) * size, funded: {}, deferred: 0, deferredBy: {} });
    ((wants && wants.hourly) || []).forEach(function (r) {
      var t = ms(r.h);
      if (t === null || t < startMs || t >= endMs) return;
      var b = buckets[Math.floor((t - startMs) / size)];
      var fam = top.indexOf(r.family) >= 0 ? r.family : OTHER;
      b.funded[fam] = (b.funded[fam] || 0) + (num(r.funded) || 0);
      var d = num(r.deferred) || 0;
      b.deferred += d;
      if (d) b.deferredBy[r.family] = (b.deferredBy[r.family] || 0) + d;
    });
    return { families: top.concat([OTHER]), buckets: buckets };
  }

  // ---------- brushing + decisions (H2) ----------
  function brushWindow(f0, f1, startMs, endMs) {
    var a = Math.max(0, Math.min(1, Math.min(f0, f1)));
    var b = Math.max(0, Math.min(1, Math.max(f0, f1)));
    var span = endMs - startMs;
    return { from: Math.round(startMs + a * span), to: Math.round(startMs + b * span) };
  }

  // Exactly the decisions inside [from, to), matching outcome and the family search.
  function filterDecisions(decisions, f) {
    f = f || {};
    var q = String(f.query || "").trim().toLowerCase();
    return (decisions || []).filter(function (d) {
      if (f.outcome && f.outcome !== "all" && d.outcome !== f.outcome) return false;
      if (q && String(d.family || "").toLowerCase().indexOf(q) < 0) return false;
      var hasFrom = typeof f.from === "number", hasTo = typeof f.to === "number";
      if (hasFrom || hasTo) {
        var t = ms(d.decided_at);
        if (t === null) return false;
        if (hasFrom && t < f.from) return false;
        if (hasTo && t >= f.to) return false;
      }
      return true;
    });
  }

  function reasonWords(d) {
    if (d.outcome === "funded") {
      return d.reason === "exploration_floor" ? "funded: exploration floor (a random control arm)" : "funded: value beat the prices";
    }
    if (d.reason === "sated") return "deferred: not hungry yet";
    if (d.reason === "capacity_reserved") return "deferred: " + (d.binding_constraint || "a lane") + " is full";
    if (d.reason === "price") return "deferred: " + (d.binding_constraint || "a limit") + " costs more than the want is worth now";
    if (d.reason === "no_acceptable_lane") return "deferred: no lane could take it";
    return "deferred: " + statusWords(d.reason);
  }

  function blockedBy(d) {
    if (d.outcome === "funded") return "—";
    if (d.reason === "sated") return "not hungry";
    return d.binding_constraint || statusWords(d.reason);
  }

  function decisionLife(d) {
    return [
      { text: "want opened · " + d.family + " asked to " + d.verb, done: true },
      { text: "bid · " + reasonWords(d), done: true },
      { text: d.outcome === "funded" ? "act would run (shadow: the old gate decided)" : "act waits for its window (shadow: the old gate decided)", done: false },
      { text: "graded · not in this report yet", done: false },
    ];
  }

  function pricesList(prices) {
    return Object.keys(prices || {}).map(function (k) { return { k: k, v: num(prices[k]) }; })
      .filter(function (p) { return p.v !== null && p.v > 0; })
      .sort(function (a, b) { return b.v - a.v || (a.k < b.k ? -1 : 1); });
  }

  // ---------- state ----------
  var state = {
    window: "24h", report: null, err: null, seq: 0, tz: undefined,
    outcome: "all", query: "", brush: null, sel: null, drag: null,
  };

  function win() {
    var w = (state.report && state.report.window) || {};
    var s = ms(w.start), e = ms(w.end);
    if (s === null || e === null || e <= s) {
      e = Date.now();
      s = e - { "24h": 24, "7d": 168, "30d": 720 }[state.window] * HOUR_MS;
    }
    return { start: s, end: e };
  }

  function sectionData(key) {
    return state.report ? state.report[key] : undefined;
  }

  function setBody(doc, id, html) {
    var el = doc.getElementById(id);
    if (el) el.innerHTML = html;
  }

  function empty(key) {
    return emptyHtml(emptyText(state.report, key, state.err, state.tz));
  }

  // ---------- renderers ----------
  function renderTiles(doc) {
    var limits = sectionData("limits") || [];
    var tiles = limits.filter(function (l) { return l && l.now; }).map(function (l, i) { return { l: l, i: i }; })
      .sort(function (a, b) {
        return (STATE_RANK[a.l.now.bind_state] ?? 9) - (STATE_RANK[b.l.now.bind_state] ?? 9) || a.i - b.i;
      }).slice(0, 4).map(function (x) {
        var l = x.l;
        return '<div class="tile" data-tile="' + esc(l.constraint) + '"><div class="tile-label">' + esc(l.label || l.constraint) +
          '</div><div class="tile-value">' + esc(fmtLevel(l.now.level, l.unit)) + '</div><div class="tile-note">' +
          stateChip(l.now.bind_state) + "<span>line " + esc(fmtLevel(l.line, l.unit)) + "</span></div></div>";
      });
    if (!tiles.length) tiles.push('<div class="tile">' + empty("limits") + "</div>");
    tiles.push(servicesTile());
    setBody(doc, "beTiles", tiles.join(""));
  }

  function servicesTile() {
    var s = sectionData("services");
    var head = '<div class="tile" data-tile="services"><div class="tile-label">Services that should run</div>';
    if (!s || !Array.isArray(s.down)) return head + empty("services") + "</div>";
    var checked = s.checked_at ? "checked " + fmtWhen(s.checked_at, state.tz) : "";
    if (!s.down.length) {
      return head + '<div class="tile-value quiet">all running</div><div class="tile-note"><span>' + esc(checked) + "</span></div></div>";
    }
    var rows = s.down.map(function (d) {
      return '<li data-down="' + esc(d.name) + '"><strong>' + esc(d.name) + "</strong> on " + esc(d.host || "?") + " · " +
        esc(d.state || "stopped") + (d.since ? " since " + esc(fmtWhen(d.since, state.tz)) : "") + "</li>";
    }).join("");
    return head + '<div class="tile-value">' + s.down.length + " stopped</div>" + chip("needs a look", "warn") +
      '<ul class="down-list">' + rows + '</ul><div class="tile-note"><span>' + esc(checked) + "</span></div></div>";
  }

  function axisHtml(w) {
    var marks = [0, 0.25, 0.5, 0.75, 1].map(function (f) {
      return "<span>" + esc(fmtTime(new Date(w.start + f * (w.end - w.start)).toISOString(),
        { weekday: "short", hour: "2-digit", minute: "2-digit" }, state.tz)) + "</span>";
    });
    return marks.join("");
  }

  function limitRow(l, w) {
    var now = l.now || {};
    return '<div class="strip" data-limit="' + esc(l.constraint) + '"><div class="strip-name"><div class="strong">' +
      esc(l.label || l.constraint) + '</div><div class="muted small">line ' + esc(fmtLevel(l.line, l.unit)) +
      '</div></div><div class="track" data-brush="1" data-limit-track="' + esc(l.constraint) + '">' + limitSvg(l, w.start, w.end) +
      '<div class="brush hidden"></div></div><div class="strip-now"><div class="strong">' + esc(fmtLevel(now.level, l.unit)) +
      '</div><div class="muted small">' + esc(STATE_WORDS[now.bind_state] || now.bind_state || "") + "</div></div></div>";
  }

  function renderLimits(doc) {
    var limits = sectionData("limits") || [];
    var w = win();
    if (!limits.length) {
      setBody(doc, "beLimitRows", empty("limits"));
      setBody(doc, "beLimitAxis", "");
      setBody(doc, "beLimitIdle", "");
      return;
    }
    var active = limits.filter(function (l) { return !isIdleLimit(l); });
    var idle = limits.filter(isIdleLimit);
    setBody(doc, "beLimitRows", active.length ? active.map(function (l) { return limitRow(l, w); }).join("")
      : emptyHtml("Every lane had room to spare every hour in this window."));
    setBody(doc, "beLimitAxis", axisHtml(w));
    setBody(doc, "beLimitIdle", idle.length
      ? "<details><summary>" + idle.length + " idle lane" + (idle.length === 1 ? "" : "s") + " folded away (" +
        esc(idle.map(function (l) { return l.label || l.constraint; }).join(", ")) +
        "): room to spare every hour.</summary>" + idle.map(function (l) { return limitRow(l, w); }).join("") + "</details>"
      : "");
    paintBrush(doc);
  }

  function renderWants(doc) {
    var wants = sectionData("wants");
    var w = win();
    if (!wants || !(wants.hourly || []).length) {
      setBody(doc, "beWantsLegend", "");
      setBody(doc, "beWantsChart", empty("wants"));
      setBody(doc, "beWantsAxis", "");
      return;
    }
    var bh = WANT_BUCKET_HOURS[state.window] || 1;
    var b = bucketWants(wants, w.start, w.end, bh);
    var max = 0;
    b.buckets.forEach(function (k) {
      var s = k.deferred;
      Object.keys(k.funded).forEach(function (f) { s += k.funded[f]; });
      max = Math.max(max, s);
    });
    max = max || 1;
    setBody(doc, "beWantsLegend", b.families.map(function (f) {
      return '<span class="key"><span class="sw" style="background:' + colorForFamily(f) + '"></span>' + esc(f === OTHER ? "other families" : f) + "</span>";
    }).join("") + '<span class="key"><span class="sw ghost"></span>deferred (wanted, not funded)</span>');
    var bars = b.buckets.map(function (k) {
      var segs = "";
      var funded = 0;
      var parts = [];
      b.families.forEach(function (f) {
        var v = k.funded[f] || 0;
        if (!v) return;
        funded += v;
        parts.push(f + " " + v);
        segs += '<div class="seg" style="height:' + (v / max * 100).toFixed(2) + "%;background:" + colorForFamily(f) + '"></div>';
      });
      if (k.deferred) segs += '<div class="seg ghost" style="height:' + (k.deferred / max * 100).toFixed(2) + '%"></div>';
      var deferredParts = Object.keys(k.deferredBy).map(function (f) { return f + " " + k.deferredBy[f]; });
      var tip = fmtTime(new Date(k.start).toISOString(), { weekday: "short", hour: "2-digit", minute: "2-digit" }, state.tz) +
        " (" + bh + " h) · funded " + funded + (parts.length ? " (" + parts.join(", ") + ")" : "") + " · deferred " + k.deferred +
        (deferredParts.length ? " (" + deferredParts.join(", ") + ")" : "");
      return '<div class="bar" data-tip="' + esc(tip) + '">' + segs + "</div>";
    }).join("");
    setBody(doc, "beWantsChart", '<div class="bars" data-brush="1">' + bars + '<div class="brush hidden"></div></div>');
    setBody(doc, "beWantsAxis", axisHtml(w));
    paintBrush(doc);
  }

  function renderRhythm(doc) {
    var rows = sectionData("rhythm") || [];
    if (!rows.length) { setBody(doc, "beRhythm", empty("rhythm")); return; }
    var days = {};
    rows.forEach(function (r) { (days[r.day] = days[r.day] || {})[r.hour] = r; });
    var html = Object.keys(days).sort().map(function (day) {
      var cells = "";
      for (var h = 0; h < 24; h++) {
        var r = days[day][h];
        var v = r ? (num(r.heavy) || 0) : 0;
        var tip = day + " " + String(h).padStart(2, "0") + ":00 · " + (r ? v + " heavy of " + (num(r.total) || 0) + " acts" : "nothing recorded");
        cells += '<div class="cell" style="background:' + RAMP[Math.min(v, 6)] + '" data-tip="' + esc(tip) + '"></div>';
      }
      return '<div class="hm-row"><span class="hm-day">' + esc(day.slice(5)) + "</span>" + cells + "</div>";
    }).join("");
    html += '<div class="hm-axis"><span>00</span><span>06</span><span>12</span><span>18</span></div>' +
      '<div class="ramp"><span>0</span>' + RAMP.slice(1).map(function (c) { return '<span class="sw" style="background:' + c + '"></span>'; }).join("") +
      "<span>6+ heavy acts</span></div>";
    setBody(doc, "beRhythm", html);
  }

  function renderMoney(doc) {
    var m = sectionData("money");
    if (!m || num(m.mtd_usd) === null && !(m.daily_kwh || []).length) { setBody(doc, "beMoney", empty("money")); return; }
    var figs = [["Spent so far", m.mtd_usd], ["Forecast by month end", m.forecast_usd], ["Envelope", m.envelope_usd]]
      .map(function (p) { return '<div><div class="muted small">' + p[0] + '</div><div class="big">' + esc(fmtUsd(p[1])) + "</div></div>"; }).join("");
    var note = num(m.floor_usd) !== null
      ? '<p class="small">The always-on floor alone reaches ' + esc(fmtUsd(m.floor_usd)) + " this month" +
        (num(m.envelope_usd) !== null && m.floor_usd > m.envelope_usd ? ", over the envelope: this limit is not Orion's doing, and is hatched in the limits chart." : ".") +
        (num(m.rate_usd_per_kwh) !== null ? " Rate: " + esc(fmtUsd(m.rate_usd_per_kwh)) + " per kWh." : "") + "</p>"
      : "";
    var days = m.daily_kwh || [];
    var max = Math.max.apply(null, days.map(function (d) { return num(d.kwh) || 0; }).concat([1]));
    var today = state.report && state.report.generated_at ? String(state.report.generated_at).slice(0, 10) : null;
    var bars = days.length ? '<div class="kwh">' + days.map(function (d) {
      var partial = d.d === today;
      return '<div class="kbar' + (partial ? " partial" : "") + '" style="height:' + ((num(d.kwh) || 0) / max * 100).toFixed(1) +
        '%" data-tip="' + esc(d.d + " · " + (num(d.kwh) === null ? "unknown" : d.kwh.toFixed(1) + " kWh") + (partial ? " (day not over)" : "")) + '"></div>';
    }).join("") + '</div><div class="axis3"><span>' + esc(days[0].d.slice(5)) +
      '</span><span><span class="sw" style="background:#3987e5"></span>kWh per day (house)</span><span>' + esc(days[days.length - 1].d.slice(5)) + "</span></div>"
      : emptyHtml(missingReason(state.report, "money") || "No daily kWh yet.");
    setBody(doc, "beMoney", '<div class="figs">' + figs + "</div>" + note + bars);
  }

  function renderUnits(doc) {
    var rows = sectionData("unit_costs") || [];
    if (!rows.length) { setBody(doc, "beUnits", empty("unit_costs")); return; }
    setBody(doc, "beUnits", '<table><thead><tr><th scope="col">Unit</th><th scope="col">Money</th><th scope="col">Energy</th>' +
      '<th scope="col">Heat</th><th scope="col">Status</th><th scope="col">Where it comes from</th></tr></thead><tbody>' +
      rows.map(function (u) {
        var wh = num(u.wh);
        return '<tr><th scope="row">' + esc(u.unit) + "</th><td>" + esc(fmtUsd(u.usd)) + "</td><td>" +
          esc(wh === null ? "not measured" : wh >= 1000 ? (wh / 1000).toFixed(2) + " kWh" : +wh.toFixed(2) + " Wh") + "</td><td>" +
          esc(num(u.heat_c) === null ? "—" : "+" + u.heat_c.toFixed(2) + " °C in the cabinet") + "</td><td>" + statusChip(u.status) +
          '</td><td class="small">' + esc(u.source || "") + "</td></tr>";
      }).join("") + "</tbody></table>");
  }

  function renderValue(doc) {
    var rows = sectionData("value_per_cost") || [];
    if (!rows.length) { setBody(doc, "beValue", empty("value_per_cost")); return; }
    setBody(doc, "beValue", '<table><thead><tr><th scope="col">Family</th><th scope="col">Value</th><th scope="col">Cost per act</th>' +
      '<th scope="col">Value per dollar</th><th scope="col">Evidence</th><th scope="col">Status</th></tr></thead><tbody>' +
      rows.map(function (r) {
        var v = num(r.value), c = num(r.usd_per_act);
        var per = v !== null && c !== null && c > 0 ? fmtCount(v / c) : "not measured";
        return '<tr data-family="' + esc(r.family) + '"><th scope="row"><span class="sw" style="background:' + colorForFamily(r.family) + '"></span>' +
          esc(r.family) + "</th><td>" + esc(v === null ? "unknown" : v.toFixed(2)) + ' <span class="muted small">' + esc(r.value_metric || "") +
          "</span></td><td>" + esc(fmtUsd(c)) + "</td><td>" + esc(per) + "</td><td>" + esc(num(r.evidence_n) === null ? "?" : r.evidence_n) +
          " acts</td><td>" + statusChip(r.status) + "</td></tr>";
      }).join("") + "</tbody></table>");
  }

  function hbar(label, value, max, text, tip, color) {
    return '<div class="hbar" data-tip="' + esc(tip) + '"><span class="hbar-label">' + label + '</span><span class="hbar-track"><span style="width:' +
      Math.min(100, (value / (max || 1)) * 100).toFixed(1) + "%;background:" + (color || "#3987e5") + '"></span></span><span class="hbar-text">' +
      esc(text) + "</span></div>";
  }

  function renderWaste(doc) {
    var w = sectionData("waste");
    var by = (w && w.failed_lease_s_by_lane) || {};
    var lanes = Object.keys(by).filter(function (k) { return num(by[k]) !== null; }).sort(function (a, b) { return by[b] - by[a]; });
    if (!lanes.length) { setBody(doc, "beWaste", empty("waste")); return; }
    var max = Math.max.apply(null, lanes.map(function (k) { return by[k]; }).concat([1]));
    var html = lanes.map(function (k) {
      return hbar(esc(k), by[k], max, fmtSeconds(by[k]), k + " · " + Math.round(by[k]) + " lease-seconds ended in failure", "#d95926");
    }).join("");
    var top = (w.top || []).map(function (t) {
      if (!t || typeof t !== "object") return "<li>" + esc(t) + "</li>";
      var s = num(t.failed_lease_s);
      return "<li>" + esc([t.family, t.lane].filter(Boolean).join(" on ")) + (s !== null ? " · " + esc(fmtSeconds(s)) : "") +
        (num(t.n) !== null ? " · " + t.n + " failed" : "") + "</li>";
    }).join("");
    setBody(doc, "beWaste", '<div class="legend"><span class="key"><span class="sw" style="background:#d95926"></span>GPU time held by leases that failed</span></div>' +
      html + (top ? '<div class="muted small">Biggest wasters</div><ul class="plain">' + top + "</ul>" : ""));
  }

  function renderQueue(doc) {
    var rows = sectionData("queue_wait") || [];
    if (!rows.length) { setBody(doc, "beQueue", empty("queue_wait")); return; }
    var max = Math.max.apply(null, rows.map(function (r) { return num(r.p90_s) || 0; }).concat([1]));
    setBody(doc, "beQueue", '<div class="legend"><span class="key"><span class="sw" style="background:#3987e5"></span>typical wait (median)</span>' +
      '<span class="key"><span class="sw" style="background:#1e3a8a"></span>slow wait (9 in 10 waited less)</span></div>' +
      rows.slice().sort(function (a, b) { return (num(b.p90_s) || 0) - (num(a.p90_s) || 0); }).map(function (r) {
        var p50 = num(r.p50_s) || 0, p90 = num(r.p90_s) || 0;
        return '<div class="hbar" data-tip="' + esc(r.lane + " · " + r.family + " · median " + fmtSeconds(r.p50_s) + " · 90th percentile " +
          fmtSeconds(r.p90_s) + " · " + r.n + " waits") + '"><span class="hbar-label">' + esc(r.lane) + ' <span class="muted">' + esc(r.family) +
          '</span></span><span class="hbar-track"><span style="width:' + (p90 / max * 100).toFixed(1) + '%;background:#1e3a8a"></span>' +
          '<span class="over" style="width:' + (p50 / max * 100).toFixed(1) + '%;background:#3987e5"></span></span><span class="hbar-text">' +
          esc(fmtSeconds(r.p50_s) + " / " + fmtSeconds(r.p90_s)) + "</span></div>";
      }).join(""));
  }

  function renderVram(doc) {
    var rows = sectionData("vram") || [];
    if (!rows.length) { setBody(doc, "beVram", empty("vram")); return; }
    setBody(doc, "beVram", '<div class="legend"><span class="key"><span class="sw" style="background:#3987e5"></span>used</span>' +
      '<span class="key"><span class="sw" style="background:#1f2937"></span>headroom</span></div>' +
      rows.map(function (r) {
        var used = num(r.used_mb), total = num(r.total_mb), head = num(r.headroom_mb);
        var label = esc(r.host + " " + r.card) + ' <span class="muted">' + esc((r.roles || []).join(", ") || "no role") + "</span>";
        var text = head === null ? "unknown" : (head / 1024).toFixed(1) + " GB free";
        return '<div data-card="' + esc(r.host + ":" + r.card) + '">' + hbar(label, used || 0, total || 1, text,
          r.host + " " + r.card + " · " + (used === null ? "?" : (used / 1024).toFixed(1)) + " of " + (total === null ? "?" : (total / 1024).toFixed(1)) +
          " GB used · roles: " + ((r.roles || []).join(", ") || "none"), head !== null && total && head / total < 0.1 ? "#d95926" : "#3987e5") + "</div>";
      }).join(""));
  }

  function renderClaude(doc) {
    var c = sectionData("claude");
    if (!c || !c.window_tokens) { setBody(doc, "beClaude", empty("claude")); return; }
    var t = c.window_tokens;
    var parts = [["Input", t.input], ["Output", t.output], ["Cache read", t.cache_read], ["Cache write", t.cache_write]];
    var max = Math.max.apply(null, parts.map(function (p) { return num(p[1]) || 0; }).concat([1]));
    var events = (c.limit_events || []).map(function (e) {
      return "<li>" + esc(fmtWhen(e.at, state.tz)) + " · " + esc(statusWords(e.kind)) + "</li>";
    }).join("");
    setBody(doc, "beClaude", '<div class="figs"><div><div class="muted small">Estimated spend</div><div class="big">' + esc(fmtUsd(c.usd_est)) +
      "</div></div><div>" + statusChip(c.status) + "</div></div>" +
      '<div class="legend"><span class="key"><span class="sw" style="background:#9085e9"></span>tokens in this window</span></div>' +
      parts.map(function (p) {
        return hbar(p[0], num(p[1]) || 0, max, fmtCount(p[1]), p[0] + " · " + (num(p[1]) === null ? "unknown" : p[1].toLocaleString("en-US")) + " tokens", "#9085e9");
      }).join("") +
      '<div class="muted small">Hit a usage limit</div>' + (events ? '<ul class="plain">' + events + "</ul>" : '<p class="small">No limit hits in this window.</p>'));
  }

  function renderSees(doc) {
    var limits = (sectionData("limits") || []).filter(function (l) { return l && l.now && l.now.why; });
    if (!limits.length) { setBody(doc, "beSees", empty("limits")); return; }
    setBody(doc, "beSees", limits.map(function (l) {
      return '<div class="why-row" data-why="' + esc(l.constraint) + '"><div class="strong">' + esc(l.label || l.constraint) +
        '</div><div class="mono">' + esc(l.now.why) + "</div></div>";
    }).join(""));
  }

  function renderDecisions(doc) {
    var all = sectionData("decisions") || [];
    doc.querySelectorAll("[data-outcome]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-outcome") === state.outcome ? "true" : "false");
    });
    var brushNote = doc.getElementById("beBrushNote");
    if (brushNote) {
      brushNote.textContent = state.brush ? "Showing " + fmtWhen(new Date(state.brush.from).toISOString(), state.tz) + " to " +
        fmtWhen(new Date(state.brush.to).toISOString(), state.tz) + " (from the selection above)." : "Drag across the limits or wants chart to pick a time window.";
    }
    var clear = doc.getElementById("beBrushClear");
    if (clear) clear.classList.toggle("hidden", !state.brush);
    if (!all.length) {
      setBody(doc, "beDecRows", "");
      setBody(doc, "beDecEmpty", empty("decisions"));
      setBody(doc, "beCount", "");
      setBody(doc, "beDetail", '<p class="muted">Nothing to select.</p>');
      return;
    }
    var shown = filterDecisions(all, { outcome: state.outcome, query: state.query,
      from: state.brush && state.brush.from, to: state.brush && state.brush.to });
    setBody(doc, "beCount", esc(shown.length + " of " + all.length + " decisions"));
    setBody(doc, "beDecEmpty", shown.length ? "" : emptyHtml('No decisions match. Clear the search, the time selection, or pick "All".'));
    setBody(doc, "beDecRows", shown.map(function (d) {
      var i = all.indexOf(d);
      return '<tr data-dec="' + i + '" data-outcome-row="' + esc(d.outcome) + '" class="' + (i === state.sel ? "sel" : "") + '"><td class="nowrap">' +
        esc(fmtWhen(d.decided_at, state.tz)) + '</td><td><button type="button" class="linkish" data-pick="' + i + '"><span class="sw" style="background:' +
        colorForFamily(d.family) + '"></span>' + esc(d.family) + "</button></td><td>" + esc(d.tier) + "</td><td>" +
        chip(d.outcome, d.outcome === "funded" ? "good" : "warn") + "</td><td>" + esc(blockedBy(d)) + '</td><td class="num">' +
        esc(num(d.net) === null ? "?" : (d.net >= 0 ? "+" : "") + d.net.toFixed(3)) + "</td></tr>";
    }).join(""));
    renderDetail(doc, all);
  }

  function renderDetail(doc, all) {
    var d = state.sel !== null ? all[state.sel] : null;
    if (!d) { setBody(doc, "beDetail", '<p class="muted">Click a row to see its life, the prices it saw and its reason, verbatim.</p>'); return; }
    var life = decisionLife(d).map(function (s) {
      return '<li class="' + (s.done ? "done" : "") + '"><span class="dot" aria-hidden="true"></span>' + esc(s.text) + "</li>";
    }).join("");
    var prices = pricesList(d.prices_seen);
    var priceHtml = prices.length ? prices.map(function (p) {
      return hbar(esc(p.k), Math.min(p.v, 1), 1, p.v.toFixed(2), p.k + " price " + p.v.toFixed(3) + (p.v >= 0.999 ? " (at the ceiling)" : ""),
        p.v >= 0.999 ? "#d95926" : "#3987e5");
    }).join("") : '<p class="small">Every limit was free (price 0) at that moment.</p>';
    var displaced = (d.displaced || []).length ? esc(d.displaced.join(", ")) : "none. Nothing is displaced while bids run in shadow.";
    setBody(doc, "beDetail", '<div class="muted small">Selected decision</div><h3 id="beDetailTitle">' + esc(d.family + " · " + d.verb) + "</h3>" +
      '<div class="small">' + chip(d.outcome, d.outcome === "funded" ? "good" : "warn") + " tier " + esc(d.tier) + " · " +
      esc(fmtWhen(d.decided_at, state.tz)) + " · reason: " + esc(d.reason) + "</div>" +
      '<div class="small" data-binding>' + (d.outcome === "deferred" ? "Blocked by: <strong>" + esc(blockedBy(d)) + "</strong>" : "Funded on its merits") + "</div>" +
      '<div class="muted small">Its life so far</div><ol class="life">' + life + "</ol>" +
      '<div class="muted small">Prices it saw (0 = free, 1 = at the ceiling). Limits priced at 0 are hidden.</div>' + priceHtml +
      '<div class="muted small">Reason, verbatim</div><div class="mono box">' + esc(d.why || "(no reason recorded)") + "</div>" +
      '<div class="small" data-displaced>Displaced: ' + displaced + "</div>");
  }

  function renderStatus(doc) {
    var el = doc.getElementById("beStatus");
    if (!el) return;
    if (state.err) { el.textContent = unreachableText(state.err, state.tz); el.className = "status warn"; return; }
    if (!state.report) { el.textContent = "loading…"; el.className = "status"; return; }
    var gen = state.report.generated_at ? "Report from " + fmtWhen(state.report.generated_at, state.tz) + "." : "";
    var miss = (state.report.missing || []).length ? " " + state.report.missing.length + " section(s) missing; each says why." : "";
    el.textContent = gen + miss;
    el.className = "status";
  }

  function render(doc) {
    doc.querySelectorAll("[data-window]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-window") === state.window ? "true" : "false");
    });
    renderStatus(doc);
    renderTiles(doc);
    renderLimits(doc);
    renderWants(doc);
    renderRhythm(doc);
    renderMoney(doc);
    renderWaste(doc);
    renderQueue(doc);
    renderVram(doc);
    renderClaude(doc);
    renderUnits(doc);
    renderValue(doc);
    renderSees(doc);
    renderDecisions(doc);
  }

  // ---------- brushing in the DOM ----------
  function paintBrush(doc) {
    var w = win();
    doc.querySelectorAll("[data-brush] > .brush").forEach(function (el) {
      var b = state.drag ? brushWindow(state.drag.f0, state.drag.f1, w.start, w.end) : state.brush;
      if (!b) { el.classList.add("hidden"); return; }
      el.classList.remove("hidden");
      el.style.left = ((b.from - w.start) / (w.end - w.start) * 100).toFixed(2) + "%";
      el.style.width = ((b.to - b.from) / (w.end - w.start) * 100).toFixed(2) + "%";
    });
  }

  function fracIn(el, clientX) {
    var r = el.getBoundingClientRect();
    return r.width ? (clientX - r.left) / r.width : 0;
  }

  function wireBrush(doc) {
    doc.addEventListener("mousedown", function (ev) {
      var el = ev.target.closest && ev.target.closest("[data-brush]");
      if (!el || ev.button !== 0) return;
      ev.preventDefault();
      var f = fracIn(el, ev.clientX);
      state.drag = { el: el, f0: f, f1: f };
      paintBrush(doc);
    });
    doc.addEventListener("mousemove", function (ev) {
      if (!state.drag) return;
      state.drag.f1 = fracIn(state.drag.el, ev.clientX);
      paintBrush(doc);
    });
    doc.addEventListener("mouseup", function () {
      if (!state.drag) return;
      var d = state.drag;
      state.drag = null;
      if (Math.abs(d.f1 - d.f0) > 0.005) {
        var w = win();
        state.brush = brushWindow(d.f0, d.f1, w.start, w.end);
        state.sel = null;
      }
      paintBrush(doc);
      renderDecisions(doc);
    });
  }

  // ---------- tooltips ----------
  function wireTooltips(doc) {
    var tip = doc.getElementById("beTip");
    if (!tip) return;
    function show(text, ev) {
      tip.textContent = text;
      tip.classList.remove("hidden");
      var x = Math.min(ev.clientX + 14, (doc.defaultView.innerWidth || 1200) - 300);
      tip.style.left = Math.max(4, x) + "px";
      tip.style.top = (ev.clientY + 14) + "px";
    }
    doc.addEventListener("mousemove", function (ev) {
      var t = ev.target;
      var track = t.closest && t.closest("[data-limit-track]");
      if (track) {
        var key = track.getAttribute("data-limit-track");
        var limit = (sectionData("limits") || []).filter(function (l) { return l.constraint === key; })[0];
        var w = win();
        var text = limit && limitTip(limit, pointAt(limit, fracIn(track, ev.clientX), w.start, w.end), state.tz);
        if (text) { show(text, ev); return; }
      }
      var host = t.closest && t.closest("[data-tip]");
      if (host) { show(host.getAttribute("data-tip"), ev); return; }
      tip.classList.add("hidden");
    });
    doc.addEventListener("mouseleave", function () { tip.classList.add("hidden"); });
  }

  // ---------- loading ----------
  function load(doc, windowKey) {
    if (WINDOWS.indexOf(windowKey) >= 0) state.window = windowKey;
    state.brush = null;
    state.sel = null;
    var seq = ++state.seq;
    renderStatus(doc);
    var fetchFn = (doc.defaultView && doc.defaultView.fetch) || fetch;
    return fetchFn("/api/body-economy/report?window=" + encodeURIComponent(state.window) + "&limit=500", { cache: "no-store" })
      .then(function (resp) {
        return resp.json().catch(function () { return {}; }).then(function (body) { return { resp: resp, body: body }; });
      })
      .then(function (r) {
        if (seq !== state.seq) return;
        if (!r.resp.ok) {
          state.err = { status: r.resp.status, detail: r.body && r.body.detail, unreachable_since: r.body && r.body.unreachable_since };
          state.report = null;
        } else {
          state.err = null;
          state.report = r.body;
        }
        render(doc);
      })
      .catch(function (e) {
        if (seq !== state.seq) return;
        state.err = { network: true, detail: "the Hub did not answer (" + (e && e.message || e) + ")" };
        state.report = null;
        render(doc);
      });
  }

  function init(doc) {
    doc.querySelectorAll("[data-window]").forEach(function (b) {
      b.addEventListener("click", function () { load(doc, b.getAttribute("data-window")); });
    });
    doc.querySelectorAll("[data-outcome]").forEach(function (b) {
      b.addEventListener("click", function () { state.outcome = b.getAttribute("data-outcome"); renderDecisions(doc); });
    });
    var q = doc.getElementById("beWhyWait");
    if (q) q.addEventListener("input", function () { state.query = q.value; renderDecisions(doc); });
    var clear = doc.getElementById("beBrushClear");
    if (clear) clear.addEventListener("click", function () { state.brush = null; paintBrush(doc); renderDecisions(doc); });
    var rows = doc.getElementById("beDecRows");
    if (rows) rows.addEventListener("click", function (ev) {
      var tr = ev.target.closest && ev.target.closest("tr[data-dec]");
      if (!tr) return;
      state.sel = Number(tr.getAttribute("data-dec"));
      renderDecisions(doc);
    });
    wireBrush(doc);
    wireTooltips(doc);
    var resizeTimer = null;
    doc.defaultView.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { if (state.report) render(doc); }, 150);
    });
    return load(doc, state.window);
  }

  var api = {
    WINDOWS: WINDOWS, STATE_WORDS: STATE_WORDS, colorForFamily: colorForFamily, fmtLevel: fmtLevel, fmtUsd: fmtUsd,
    fmtSeconds: fmtSeconds, fmtCount: fmtCount, fmtWhen: fmtWhen, isIdleLimit: isIdleLimit, limitRange: limitRange,
    limitSvg: limitSvg, pointAt: pointAt, limitTip: limitTip, topFamilies: topFamilies, bucketWants: bucketWants,
    brushWindow: brushWindow, filterDecisions: filterDecisions, reasonWords: reasonWords, blockedBy: blockedBy,
    decisionLife: decisionLife, pricesList: pricesList, missingReason: missingReason, unreachableText: unreachableText,
    emptyText: emptyText, esc: esc, state: state, render: render, init: init, load: load,
  };
  if (typeof window !== "undefined") window.BodyEconomyTab = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();

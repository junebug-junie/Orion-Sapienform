// Orion energy tab (static/orion-energy.html): Orion's own kWh by source, stacked,
// against the whole house from Rocky Mountain Power, plus Orion's share of the house.
// Reads GET /api/energy/orion. Unknown stays "unknown": a missing bucket is a gap, never 0.
(function () {
  "use strict";

  // Fixed per source, never by rank (dataviz reference palette, dark steps, validated).
  var SOURCE_COLORS = {
    athena: "var(--series-1)",
    circe: "var(--series-2)",
    hecate: "var(--series-3)",
    cooling: "var(--series-4)",
  };
  var EXTRA_COLORS = ["var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)"];
  var SOURCE_NAMES = { cooling: "cabinet cooling" };
  var GRAINS = ["hour", "day", "week", "month", "quarter", "year"];

  function colorFor(source, sources) {
    if (SOURCE_COLORS[source]) return SOURCE_COLORS[source];
    var extras = (sources || []).filter(function (s) { return !SOURCE_COLORS[s]; });
    var i = extras.indexOf(source);
    return i >= 0 && i < EXTRA_COLORS.length ? EXTRA_COLORS[i] : "var(--ink-muted)";
  }

  function sourceName(source) {
    return SOURCE_NAMES[source] || source;
  }

  function fmtKwh(v) {
    if (v === null || v === undefined) return "unknown";
    if (Math.abs(v) >= 100) return Math.round(v).toLocaleString("en-US") + " kWh";
    return v.toFixed(Math.abs(v) >= 10 ? 1 : 2) + " kWh";
  }

  function fmtUsd(v) {
    if (v === null || v === undefined) return "unknown";
    return (v < 0 ? "-$" : "$") + Math.abs(v).toFixed(2);
  }

  function fmtPct(v) {
    if (v === null || v === undefined) return "unknown";
    return (v * 100).toFixed(1) + "%";
  }

  function niceMax(v) {
    if (!(v > 0)) return 1;
    var mag = Math.pow(10, Math.floor(Math.log10(v)));
    var steps = [1, 2, 2.5, 5, 10];
    for (var i = 0; i < steps.length; i++) {
      if (steps[i] * mag >= v) return steps[i] * mag;
    }
    return 10 * mag;
  }

  // Show every k-th x label so ~70px labels never collide.
  function labelEvery(n, plotWidth) {
    if (n <= 0) return 1;
    return Math.max(1, Math.ceil((n * 70) / Math.max(plotWidth, 1)));
  }

  // House meter delivered fewer hours than the bucket has had: RMP lag or a hole.
  function housePartial(b) {
    return b.house_kwh !== null && b.house_hours < Math.floor(b.hours_elapsed) - 1;
  }

  function tooltipRows(b, sources) {
    var rows = [];
    sources.forEach(function (s) {
      var kwh = b.kwh_by_source[s];
      var covered = b.covered_hours_by_source[s];
      var note = "no data";
      if (kwh !== undefined) {
        note = b.hours_elapsed > 0 && covered !== undefined
          ? Math.min(100, Math.round((covered / b.hours_elapsed) * 100)) + "% of the time measured"
          : "";
      }
      rows.push({ swatch: colorFor(s, sources), name: sourceName(s),
                  value: kwh === undefined ? "unknown" : fmtKwh(kwh), note: note });
    });
    rows.push({ name: "Orion total", value: fmtKwh(b.orion_kwh), note: "", strong: true });
    rows.push({
      name: "House (RMP meter)", value: fmtKwh(b.house_kwh), line: true,
      note: b.house_kwh === null ? "no RMP data yet"
        : b.house_hours + " of " + Math.round(b.hours_elapsed) + " h delivered",
    });
    rows.push({
      name: "Orion share", value: fmtPct(b.orion_share),
      note: (b.overlap_hours ? "over " + b.overlap_hours + " h both meters have"
        : b.orion_kwh === null ? "no Orion data" : b.house_kwh === null ? "needs RMP data" : "no hour both meters have") +
        (b.partial_hours ? "; " + b.partial_hours + " partly measured h left out" : ""),
    });
    var delta = deltaPoint(b);
    rows.push({
      name: "Rest of the house", value: delta ? fmtKwh(delta.rest) : "unknown",
      note: !delta ? "" : delta.rest < 0
        ? "Orion read above the house: RMP rounds each hour to whole kWh"
        : "house " + fmtKwh(delta.house) + " − Orion " + fmtKwh(delta.orion) + ", same " + b.overlap_hours + " h",
    });
    rows.push({
      name: "Orion cost", value: fmtUsd(b.orion_usd),
      note: b.orion_kwh === null ? "no Orion data" : b.orion_usd === null ? "no price for these hours"
        : "priced " + fmtKwh(b.orion_priced_kwh) + " at the next-kWh rate",
    });
    rows.push({ name: "House cost", value: fmtUsd(b.house_usd), note: "pre-tax" });
    return rows;
  }

  function windowTotals(buckets) {
    var t = { orion: 0, ovO: 0, ovH: 0, ovHours: 0, houseHours: 0, orionUsd: 0, houseUsd: 0,
              hasO: false, hasOv: false, hasOU: false, hasHU: false };
    buckets.forEach(function (b) {
      if (b.orion_kwh !== null) { t.orion += b.orion_kwh; t.hasO = true; }
      t.houseHours += b.house_hours || 0;
      if (b.overlap_hours) {
        t.ovO += b.orion_kwh_overlap; t.ovH += b.house_kwh_overlap; t.ovHours += b.overlap_hours; t.hasOv = true;
      }
      if (b.orion_usd !== null) { t.orionUsd += b.orion_usd; t.hasOU = true; }
      if (b.house_usd !== null) { t.houseUsd += b.house_usd; t.hasHU = true; }
    });
    return {
      orion_kwh: t.hasO ? t.orion : null,
      // Both kWh tiles show the same hours (the share's hours), so they compare directly.
      orion_kwh_overlap: t.hasOv ? t.ovO : null,
      house_kwh_overlap: t.hasOv ? t.ovH : null,
      overlap_hours: t.ovHours,
      house_hours: t.houseHours,
      orion_share: t.hasOv && t.ovH > 0 ? t.ovO / t.ovH : null,
      orion_usd: t.hasOU ? t.orionUsd : null,
      house_usd: t.hasHU ? t.houseUsd : null,
    };
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // --- rendering (browser only) ----------------------------------------------------------

  var M = { left: 52, right: 12, top: 12, bottom: 30 };

  function topRoundedRect(x, y, w, h, r) {
    r = Math.min(r, w / 2, h);
    return "M" + x + "," + (y + h) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y +
      "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + (y + h) + "Z";
  }

  function axisAndGrid(W, H, yMax, fmt, ticks) {
    var out = "";
    for (var i = 0; i <= ticks; i++) {
      var v = (yMax * i) / ticks;
      var y = M.top + (H - M.top - M.bottom) * (1 - v / yMax);
      out += '<line x1="' + M.left + '" x2="' + (W - M.right) + '" y1="' + y + '" y2="' + y +
        '" stroke="' + (i === 0 ? "var(--axis)" : "var(--grid)") + '" stroke-width="1"/>';
      out += '<text x="' + (M.left - 6) + '" y="' + (y + 3) + '" text-anchor="end" font-size="10" fill="var(--ink-muted)">' +
        esc(fmt(v)) + "</text>";
    }
    return out;
  }

  function xLabels(buckets, W, H, band) {
    var every = labelEvery(buckets.length, W - M.left - M.right);
    var out = "";
    buckets.forEach(function (b, i) {
      if ((buckets.length - 1 - i) % every !== 0) return; // anchor on the newest bucket
      var x = M.left + band * (i + 0.5);
      // Keep the end labels inside the plot instead of clipping at the svg edge.
      var anchor = x + 35 > W ? "end" : (x - 35 < 0 ? "start" : "middle");
      if (anchor === "end") x = W - 2;
      out += '<text x="' + x + '" y="' + (H - 10) + '" text-anchor="' + anchor + '" font-size="10" fill="var(--ink-muted)">' +
        esc(b.label) + "</text>";
    });
    return out;
  }

  function hitTargets(buckets, W, H, band) {
    var out = "";
    buckets.forEach(function (b, i) {
      out += '<rect class="hit" data-i="' + i + '" x="' + (M.left + band * i) + '" y="' + M.top + '" width="' + band +
        '" height="' + (H - M.top - M.bottom) + '" fill="transparent"/>';
    });
    return out;
  }

  function renderKwhChart(el, data) {
    var W = Math.max(el.clientWidth || 800, 320), H = 300;
    var buckets = data.buckets, sources = data.sources;
    var plotH = H - M.top - M.bottom;
    var band = (W - M.left - M.right) / Math.max(buckets.length, 1);
    var barW = Math.max(2, Math.min(band * 0.7, 56));
    var yMax = niceMax(Math.max.apply(null, buckets.map(function (b) {
      return Math.max(b.orion_kwh || 0, b.house_kwh || 0);
    }).concat([0])));
    var y = function (v) { return M.top + plotH * (1 - v / yMax); };
    var svg = axisAndGrid(W, H, yMax, function (v) { return v >= 100 ? Math.round(v) : +v.toFixed(1); }, 4);

    buckets.forEach(function (b, i) {
      var x = M.left + band * i + (band - barW) / 2;
      var base = 0;
      var present = sources.filter(function (s) { return b.kwh_by_source[s] > 0; });
      present.forEach(function (s, k) {
        var v = b.kwh_by_source[s];
        var y0 = y(base), y1 = y(base + v);
        var gap = k > 0 ? 2 : 0; // 2px surface gap between stacked segments
        var h = Math.max(0, y0 - y1 - gap);
        svg += k === present.length - 1
          ? '<path d="' + topRoundedRect(x, y1, barW, h, 4) + '" fill="' + colorFor(s, sources) + '"/>'
          : '<rect x="' + x + '" y="' + y1 + '" width="' + barW + '" height="' + h + '" fill="' + colorFor(s, sources) + '"/>';
        base += v;
      });
      if (b.house_kwh !== null) {
        var hy = y(b.house_kwh);
        svg += '<line x1="' + (x - 4) + '" x2="' + (x + barW + 4) + '" y1="' + hy + '" y2="' + hy +
          '" stroke="var(--ink)" stroke-width="2"' + (housePartial(b) ? ' stroke-dasharray="4 3"' : "") + "/>";
      }
    });
    svg += xLabels(buckets, W, H, band) + hitTargets(buckets, W, H, band);
    el.innerHTML = '<svg width="' + W + '" height="' + H + '" role="img" aria-label="Orion kWh by source against the house">' + svg + "</svg>";
  }

  // Orion and the house over the same hours (the share's hours), so the gap between the two
  // lines is exactly the rest of the house. Null where those hours don't exist.
  function deltaPoint(b) {
    if (!b.overlap_hours) return null;
    return {
      house: b.house_kwh_overlap, orion: b.orion_kwh_overlap, rest: b.house_kwh_overlap - b.orion_kwh_overlap,
      // Both meters cover under 90% of the period (RMP lag, a gap): a dip there is missing hours, not
      // less use. A few hours short of a full week is not worth flagging.
      partial: b.overlap_hours < 0.9 * b.hours_elapsed,
    };
  }

  function renderDeltaChart(el, data) {
    var W = Math.max(el.clientWidth || 800, 320), H = 220;
    var buckets = data.buckets;
    var plotH = H - M.top - M.bottom;
    var band = (W - M.left - M.right) / Math.max(buckets.length, 1);
    var points = buckets.map(deltaPoint);
    // Orion can read above the house in an hour (RMP rounds to whole kWh), so scale to both.
    var yMax = niceMax(Math.max.apply(null, points.map(function (p) { return p ? Math.max(p.house, p.orion) : 0; }).concat([0])));
    var y = function (v) { return M.top + plotH * (1 - Math.max(v, 0) / yMax); };
    var xAt = function (i) { return M.left + band * (i + 0.5); };
    var svg = axisAndGrid(W, H, yMax, function (v) { return v >= 100 ? Math.round(v) : +v.toFixed(1); }, 4);
    // Contiguous runs of known points: a missing period breaks both lines and the gap fill.
    var runs = [], run = [];
    points.forEach(function (p, i) {
      if (p) { run.push(i); } else if (run.length) { runs.push(run); run = []; }
    });
    if (run.length) runs.push(run);
    var gap = "", lines = "", dots = "";
    runs.forEach(function (r) {
      if (r.length === 1) {
        var i0 = r[0];
        gap += '<line x1="' + xAt(i0) + '" x2="' + xAt(i0) + '" y1="' + y(points[i0].house) + '" y2="' + y(Math.min(points[i0].orion, points[i0].house)) +
          '" stroke="var(--rest-fill)" stroke-width="10"/>';
      } else {
        var top = r.map(function (i) { return xAt(i) + "," + y(points[i].house); });
        // Never shade a negative "rest": where Orion reads above the house the fill closes to zero.
        var bottom = r.slice().reverse().map(function (i) { return xAt(i) + "," + y(Math.min(points[i].orion, points[i].house)); });
        gap += '<polygon points="' + top.concat(bottom).join(" ") + '" fill="var(--rest-fill)"/>';
      }
      ["house", "orion"].forEach(function (key) {
        var color = key === "house" ? "var(--ink)" : "var(--orion-total)";
        for (var k = 1; k < r.length; k++) {
          var a = r[k - 1], c = r[k];
          lines += '<line x1="' + xAt(a) + '" y1="' + y(points[a][key]) + '" x2="' + xAt(c) + '" y2="' + y(points[c][key]) +
            '" stroke="' + color + '" stroke-width="2"' +
            (points[a].partial || points[c].partial ? ' stroke-dasharray="4 3"' : "") + "/>";
        }
        r.forEach(function (i) {
          var hollow = points[i].partial;
          dots += '<circle cx="' + xAt(i) + '" cy="' + y(points[i][key]) + '" r="4" fill="' + (hollow ? "var(--surface)" : color) +
            '" stroke="' + (hollow ? color : "var(--surface)") + '" stroke-width="2"/>';
        });
      });
    });
    svg += gap + lines + dots + xLabels(buckets, W, H, band) + hitTargets(buckets, W, H, band);
    el.innerHTML = '<svg width="' + W + '" height="' + H + '" role="img" aria-label="Orion against the whole house, same hours, kWh">' + svg + "</svg>";
  }

  function renderDeltaLegend(el) {
    el.innerHTML = [
      '<span class="flex items-center gap-1.5"><span class="inline-block w-4 border-t-2" style="border-color:var(--ink)"></span>whole house</span>',
      '<span class="flex items-center gap-1.5"><span class="inline-block w-4 border-t-2" style="border-color:var(--orion-total)"></span>Orion (all sources)</span>',
      '<span class="flex items-center gap-1.5"><span class="inline-block w-3 h-3 rounded-sm" style="background:var(--rest-fill)"></span>rest of the house</span>',
      '<span class="flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full border-2" style="border-color:var(--ink-2)"></span>hollow: both meters cover only part of that period</span>',
    ].join("");
  }

  function renderLegend(el, sources) {
    var items = sources.map(function (s) {
      return '<span class="flex items-center gap-1.5"><span class="inline-block w-3 h-3 rounded-sm" style="background:' +
        colorFor(s, sources) + '"></span>' + esc(sourceName(s)) + "</span>";
    });
    items.push('<span class="flex items-center gap-1.5"><span class="inline-block w-4 border-t-2" style="border-color:var(--ink)"></span>house (RMP meter)</span>');
    items.push('<span class="flex items-center gap-1.5"><span class="inline-block w-4 border-t-2 border-dashed" style="border-color:var(--ink)"></span>house, RMP data still arriving</span>');
    el.innerHTML = items.join("");
  }

  function renderTiles(doc, buckets) {
    var t = windowTotals(buckets);
    doc.getElementById("oeOrionKwh").textContent = fmtKwh(t.orion_kwh_overlap);
    doc.getElementById("oeOrionKwhNote").textContent = t.orion_kwh === null ? ""
      : fmtKwh(t.orion_kwh) + " incl. hours RMP hasn't sent";
    doc.getElementById("oeHouseKwh").textContent = fmtKwh(t.house_kwh_overlap);
    doc.getElementById("oeHouseKwhNote").textContent = "same " + t.overlap_hours + " h as the share";
    doc.getElementById("oeShare").textContent = fmtPct(t.orion_share);
    doc.getElementById("oeOrionUsd").textContent = fmtUsd(t.orion_usd);
    doc.getElementById("oeHouseUsd").textContent = fmtUsd(t.house_usd);
    doc.getElementById("oeHouseUsdNote").textContent = t.house_hours + " h of RMP data, pre-tax";
  }

  function renderTable(el, data) {
    var head = ["period"].concat(data.sources.map(sourceName), ["Orion", "house", "Orion share", "Orion $", "house $"]);
    var rows = data.buckets.slice().reverse().map(function (b) {
      var cells = [b.label + (b.complete ? "" : " (so far)")]
        .concat(data.sources.map(function (s) { return fmtKwh(b.kwh_by_source[s] === undefined ? null : b.kwh_by_source[s]); }))
        .concat([fmtKwh(b.orion_kwh), fmtKwh(b.house_kwh), fmtPct(b.orion_share), fmtUsd(b.orion_usd), fmtUsd(b.house_usd)]);
      return "<tr>" + cells.map(function (c, i) {
        return '<td class="px-2 py-1 ' + (i ? "text-right font-mono" : "") + '">' + esc(c) + "</td>";
      }).join("") + "</tr>";
    });
    el.innerHTML = '<table class="w-full text-xs"><thead><tr>' + head.map(function (h, i) {
      return '<th class="px-2 py-1 font-medium ' + (i ? "text-right" : "text-left") + '">' + esc(h) + "</th>";
    }).join("") + "</tr></thead><tbody>" + rows.join("") + "</tbody></table>";
  }

  function tooltipHtml(b, sources) {
    return '<div class="font-semibold mb-1" style="color:var(--ink)">' + esc(b.label) + (b.complete ? "" : " (so far)") + "</div>" +
      tooltipRows(b, sources).map(function (r) {
        var mark = r.swatch ? '<span class="inline-block w-2.5 h-2.5 rounded-sm mr-1.5" style="background:' + r.swatch + '"></span>'
          : '<span class="inline-block w-2.5 mr-1.5"></span>';
        return '<div class="flex items-baseline gap-3"><span class="flex items-center' + (r.strong ? " font-semibold" : "") + '">' + mark + esc(r.name) +
          '</span><span class="ml-auto font-mono' + (r.strong ? " font-semibold" : "") + '" style="color:var(--ink)">' + esc(r.value) + "</span></div>" +
          (r.note ? '<div class="pl-4 text-[10px]" style="color:var(--ink-muted)">' + esc(r.note) + "</div>" : "");
      }).join("");
  }

  function wireHover(doc, chartEls, data) {
    var tip = doc.getElementById("oeTooltip");
    chartEls.forEach(function (el) {
      el.onmousemove = function (ev) {
        var t = ev.target;
        if (!t || !t.classList || !t.classList.contains("hit")) { tip.classList.add("hidden"); return; }
        var b = data.buckets[+t.getAttribute("data-i")];
        tip.innerHTML = tooltipHtml(b, data.sources);
        tip.classList.remove("hidden");
        var x = ev.clientX + 14, y = ev.clientY + 14;
        var r = tip.getBoundingClientRect();
        if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
        if (y + r.height > window.innerHeight - 8) y = Math.max(8, window.innerHeight - r.height - 8);
        tip.style.left = x + "px";
        tip.style.top = y + "px";
        chartEls.forEach(function (other) {
          other.querySelectorAll(".hit").forEach(function (h) {
            h.setAttribute("fill", h.getAttribute("data-i") === t.getAttribute("data-i") ? "rgba(255,255,255,0.05)" : "transparent");
          });
        });
      };
      el.onmouseleave = function () {
        tip.classList.add("hidden");
        chartEls.forEach(function (other) {
          other.querySelectorAll(".hit").forEach(function (h) { h.setAttribute("fill", "transparent"); });
        });
      };
    });
  }

  var state = { grain: "day", data: null, seq: 0 };

  // A failed load must not leave the previous grain's numbers looking current.
  function clearView(doc) {
    ["oeKwhChart", "oeDeltaChart", "oeDeltaLegend", "oeTable", "oeLegend", "oeOrionKwhNote", "oeHouseKwhNote", "oeHouseUsdNote"]
      .forEach(function (id) { doc.getElementById(id).innerHTML = ""; });
    ["oeOrionKwh", "oeHouseKwh", "oeShare", "oeOrionUsd", "oeHouseUsd"]
      .forEach(function (id) { doc.getElementById(id).textContent = "unknown"; });
  }

  function render(doc) {
    var data = state.data;
    var status = doc.getElementById("oeStatus");
    if (!data || !data.ok) {
      clearView(doc);
      status.textContent = "Orion energy is unavailable right now (database not reachable).";
      return;
    }
    status.textContent = data.sources.length ? "" : "No Orion energy rows yet: sql-writer fills them every 10 minutes.";
    renderTiles(doc, data.buckets);
    doc.getElementById("oeHourNote").classList.toggle("hidden", data.grain !== "hour");
    renderLegend(doc.getElementById("oeLegend"), data.sources);
    var kwh = doc.getElementById("oeKwhChart"), delta = doc.getElementById("oeDeltaChart");
    renderKwhChart(kwh, data);
    renderDeltaChart(delta, data);
    renderDeltaLegend(doc.getElementById("oeDeltaLegend"));
    renderTable(doc.getElementById("oeTable"), data);
    wireHover(doc, [kwh, delta], data);
  }

  function load(doc, grain) {
    state.grain = grain;
    var seq = ++state.seq;
    doc.querySelectorAll("[data-grain]").forEach(function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-grain") === grain ? "true" : "false");
    });
    return fetch("/api/energy/orion?grain=" + encodeURIComponent(grain))
      .then(function (r) { return r.json(); })
      .catch(function () { return { ok: false }; })
      .then(function (data) {
        if (seq !== state.seq) return; // a newer click won
        state.data = data;
        render(doc);
      });
  }

  function init(doc) {
    doc.querySelectorAll("[data-grain]").forEach(function (b) {
      b.addEventListener("click", function () { load(doc, b.getAttribute("data-grain")); });
    });
    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { if (state.data) render(doc); }, 150);
    });
    load(doc, state.grain);
  }

  var api = {
    GRAINS: GRAINS, colorFor: colorFor, sourceName: sourceName, fmtKwh: fmtKwh, fmtUsd: fmtUsd,
    fmtPct: fmtPct, niceMax: niceMax, labelEvery: labelEvery, housePartial: housePartial,
    tooltipRows: tooltipRows, deltaPoint: deltaPoint, windowTotals: windowTotals, clearView: clearView, init: init,
  };
  if (typeof window !== "undefined") {
    window.OrionEnergyTab = api;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();

// Shared hover crosshair + tooltip for Hub SVG line charts (Cabinet / Office subview).
//
// Used by cabinet-sensors.js, office-env.js and cabinet-climate-overlay.js. Exposed as
// window.OrionChartHover in the browser and as a CommonJS module under node:test, so it
// must load before those scripts.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  // Linear nearest-neighbor lookup by x-position over points pre-sorted by
  // xFn. Pure/DOM-free (unlike the rest of hover wiring) so it has direct
  // unit coverage; chart point counts are small (hundreds), so a scan is fine.
  function nearestSeriesPoint(sortedPoints, xFn, targetX) {
    var best = sortedPoints[0];
    var bestDist = Math.abs(xFn(best) - targetX);
    for (var i = 1; i < sortedPoints.length; i++) {
      var dist = Math.abs(xFn(sortedPoints[i]) - targetX);
      if (dist < bestDist) {
        bestDist = dist;
        best = sortedPoints[i];
      }
    }
    return best;
  }

  // "Tue, Oct 28 14:05:00" for an ISO timestamp, or "— —" when there is none.
  function formatHoverWhen(t) {
    var when = t ? new Date(t) : null;
    var dayText = when
      ? when.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })
      : "—";
    var timeText = when
      ? when.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      : "—";
    return dayText + " " + timeText;
  }

  // Builds the hover tooltip's text: day, time, then the reading. Pure/DOM-free
  // so it has direct unit coverage independent of attachSeriesHover's DOM wiring.
  function formatHoverTooltipText(point, options) {
    options = options || {};
    var digits = options.digits !== undefined ? options.digits : 2;
    var reading = point && typeof point.value === "number" ? point.value.toFixed(digits) : "—";
    return formatHoverWhen(point && point.t) + "  ·  " + (options.label || "value") + " " + reading;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  // The chart's viewBox size, so the crosshair spans it whatever coordinates it uses.
  function viewBoxSize(chart) {
    var box = chart.viewBox && chart.viewBox.baseVal;
    return box && box.width ? { width: box.width, height: box.height } : { width: 100, height: 40 };
  }

  // Absolutely-positioned tooltip box inside `host`; place() keeps it next to the cursor.
  function createTooltip(host) {
    if (!host.style.position) host.style.position = "relative";
    var tip = document.createElement("div");
    tip.className =
      "pointer-events-none absolute z-10 hidden whitespace-nowrap rounded border border-gray-700 bg-gray-950/95 px-2 py-1 text-[10px] font-mono text-gray-200 shadow-lg";
    host.appendChild(tip);
    return {
      node: tip,
      place: function (evt) {
        tip.classList.remove("hidden");
        var hostRect = host.getBoundingClientRect();
        var left = evt.clientX - hostRect.left + 12;
        var top = evt.clientY - hostRect.top - 12;
        var tipWidth = tip.offsetWidth || 140;
        if (left + tipWidth > hostRect.width) left = evt.clientX - hostRect.left - tipWidth - 12;
        tip.style.left = Math.max(0, left) + "px";
        tip.style.top = Math.max(0, top) + "px";
      },
      hide: function () {
        tip.classList.add("hidden");
      },
    };
  }

  // Wire a mouse-hover crosshair + tooltip onto a rendered line chart.
  // `xFn`/`yFn` are the same coordinate functions used to draw the line, so the
  // hover marker lands exactly on the plotted point. Points are {t, value}.
  function attachSeriesHover(host, chart, points, xFn, yFn, options) {
    if (!points.length) return;
    options = options || {};
    var size = viewBoxSize(chart);

    var guide = svgEl("line", {
      x1: 0,
      y1: 0,
      x2: 0,
      y2: size.height,
      stroke: "#9ca3af",
      "stroke-width": 0.6,
      "vector-effect": "non-scaling-stroke",
      opacity: 0,
    });
    var dot = svgEl("circle", {
      r: options.dotRadius || 1.8,
      fill: options.color || "#818cf8",
      stroke: "#0b0f19",
      "stroke-width": 0.6,
      opacity: 0,
    });
    chart.appendChild(guide);
    chart.appendChild(dot);
    var tip = createTooltip(host);

    var sorted = points.slice().sort(function (a, b) {
      return xFn(a) - xFn(b);
    });

    function show(evt) {
      var chartRect = chart.getBoundingClientRect();
      if (!chartRect.width) return;
      var xFrac = (evt.clientX - chartRect.left) / chartRect.width;
      if (xFrac < 0 || xFrac > 1) {
        hide();
        return;
      }
      var point = nearestSeriesPoint(sorted, xFn, xFrac * size.width);
      var px = xFn(point);
      guide.setAttribute("x1", px);
      guide.setAttribute("x2", px);
      guide.setAttribute("opacity", 1);
      dot.setAttribute("cx", px);
      dot.setAttribute("cy", yFn(point.value));
      dot.setAttribute("opacity", 1);
      tip.node.textContent = formatHoverTooltipText(point, options);
      tip.place(evt);
    }

    function hide() {
      guide.setAttribute("opacity", 0);
      dot.setAttribute("opacity", 0);
      tip.hide();
    }

    chart.addEventListener("mousemove", show);
    chart.addEventListener("mouseleave", hide);
  }

  var api = {
    nearestSeriesPoint: nearestSeriesPoint,
    formatHoverWhen: formatHoverWhen,
    formatHoverTooltipText: formatHoverTooltipText,
    createTooltip: createTooltip,
    attachSeriesHover: attachSeriesHover,
  };
  if (typeof window !== "undefined") window.OrionChartHover = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();

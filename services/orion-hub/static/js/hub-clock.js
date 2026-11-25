(function (global) {
  // Hub clock (2026-11-25, Juniper: "i need a damn conversion of just simple
  // time like 12:07am UTC / 5:07 MST"). Under the UI version label, top right:
  // the time now in UTC and in Denver (MST in winter, MDT in summer; the
  // browser's own time zone data picks the right one). Hosts log in UTC.

  const ZONE = "America/Denver";

  function part(parts, type) {
    const p = parts.find(function (x) { return x.type === type; });
    return p ? p.value : "";
  }

  function clockTime(date, timeZone) {
    const parts = new Intl.DateTimeFormat("en-US", {
      hour: "numeric", minute: "2-digit", hour12: true, timeZone: timeZone, timeZoneName: "short",
    }).formatToParts(date);
    const ampm = part(parts, "dayPeriod").toLowerCase();
    const zone = timeZone === "UTC" ? "UTC" : part(parts, "timeZoneName");
    return part(parts, "hour") + ":" + part(parts, "minute") + ampm + " " + zone;
  }

  function clockText(date) {
    return clockTime(date, "UTC") + " / " + clockTime(date, ZONE);
  }

  function mount(doc) {
    const el = doc.getElementById("hubClock");
    if (!el) return;
    const tick = function () { el.textContent = clockText(new Date()); };
    tick();
    // Every 15 s: the minute turns over on screen within 15 s, at almost no cost.
    global.setInterval(tick, 15000);
  }

  const api = { clockText: clockText, clockTime: clockTime };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.OrionHubClock = api;
  if (global.document && global.document.addEventListener) {
    global.document.addEventListener("DOMContentLoaded", function () { mount(global.document); });
  }
})(typeof window !== "undefined" ? window : globalThis);

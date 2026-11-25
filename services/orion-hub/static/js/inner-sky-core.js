(function (global) {
  // Pure logic for the Inner Sky page (static/inner-sky.html). No DOM, no WebGL: tested by inner-sky-core.test.js.

  // Insert into an array kept sorted by key(item); equal keys keep arrival order. Returns the index used.
  function insertSorted(arr, item, key) {
    var k = key(item), lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (key(arr[mid]) <= k) lo = mid + 1; else hi = mid;
    }
    arr.splice(lo, 0, item);
    return lo;
  }

  // Message rates from two readings of cumulative counts. prev: {key: {n, t}} from the last call (or null).
  // A key seen for the first time, or whose count went down (the counter was reset), has no rate yet:
  // it is left out, never guessed.
  function countRates(prev, items, keyOf, now) {
    var next = {}, rates = {};
    for (var i = 0; i < items.length; i++) {
      var it = items[i], key = keyOf(it), p = prev ? prev[key] : null;
      next[key] = { n: it.n, t: now };
      if (p && now > p.t && it.n >= p.n) rates[key] = (it.n - p.n) / (now - p.t);
    }
    return { rates: rates, next: next };
  }

  // Pulses drawn per second along a link carrying `rate` real messages per second (log scale, capped).
  function pulsesPerSecond(rate) {
    return rate > 0 ? Math.min(3, 0.15 * Math.log2(1 + rate)) : 0;
  }

  // Events per bin over [t0, t1). Events outside the range are not counted.
  function binCounts(events, t0, t1, nBins) {
    var bins = new Array(nBins).fill(0), span = (t1 - t0) / nBins;
    for (var i = 0; i < events.length; i++) {
      var b = Math.floor((events[i].t - t0) / span);
      if (b >= 0 && b < nBins) bins[b]++;
    }
    return bins;
  }

  // A service is live when it published within `liveSec` of the server's own clock.
  function isLive(service, serverNow, liveSec) {
    return service.last != null && serverNow - service.last < liveSec;
  }

  var api = { insertSorted: insertSorted, countRates: countRates, pulsesPerSecond: pulsesPerSecond, binCounts: binCounts, isLive: isLive };
  global.OrionInnerSkyCore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);

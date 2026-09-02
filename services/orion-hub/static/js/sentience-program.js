/* Sentience Striving Program instrument board.
 *
 * Renders /api/sentience-program. Holds no program knowledge of its own: every
 * fact, including which outcome an instrument ladders to and whether a recorded
 * claim still holds, is decided server-side by the shared reducer. This file
 * only decides layout, so the page cannot drift from the CI gate.
 */
(function () {
  "use strict";

  var state = null;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function fmtHours(h) {
    if (h === null || h === undefined) return "—";
    return h >= 48 ? (h / 24).toFixed(1) + "d" : h.toFixed(1) + "h";
  }

  function fmtAgo(iso) {
    if (!iso) return "—";
    var mins = (Date.now() - new Date(iso).getTime()) / 60000;
    if (mins < 0) return "just now";
    if (mins < 90) return Math.round(mins) + "m ago";
    if (mins < 60 * 48) return Math.round(mins / 60) + "h ago";
    return Math.round(mins / 1440) + "d ago";
  }

  function cell(key, value, note, cls) {
    var c = el("div", "cell");
    c.appendChild(el("div", "k", key));
    c.appendChild(el("div", "v" + (cls ? " " + cls : ""), value));
    if (note) c.appendChild(el("div", "note", note));
    return c;
  }

  function renderOutcomes(data) {
    var host = document.getElementById("outcomes");
    host.textContent = "";
    Object.keys(data.outcomes).sort().forEach(function (id) {
      var oc = data.outcomes[id];
      var box = el("div", "oc");
      var h = el("h3");
      h.appendChild(el("span", "id", id));
      h.appendChild(document.createTextNode(oc.title));
      box.appendChild(h);
      box.appendChild(el("p", null, oc.claim));

      // Count instruments and, separately, claims explicitly blocking this
      // outcome -- "3 instruments" reads like progress when one of them is
      // blocked, so the blocker gets its own line rather than being averaged in.
      var mine = data.instruments.filter(function (i) { return i.outcome === id; });
      var blockers = 0;
      data.instruments.forEach(function (i) {
        i.claims.forEach(function (c) { if (c.blocks === id) blockers += 1; });
      });
      var line = mine.length + " instrument" + (mine.length === 1 ? "" : "s");
      if (blockers) line += " · " + blockers + " recorded blocker" + (blockers === 1 ? "" : "s");
      box.appendChild(el("div", "n", line));
      host.appendChild(box);
    });
  }

  function renderInstrument(inst) {
    var broken = !inst.module_exists || inst.entrypoint_exists === false;
    var blocked = inst.claims.some(function (c) { return c.blocks; }) || inst.review_stale;
    var box = el("div", "inst" + (broken ? " broken" : blocked ? " blocked" : ""));

    var head = el("div", "head");
    var left = el("div");
    left.appendChild(el("div", "title", inst.title));
    left.appendChild(el("div", "theory", inst.theory + " · " + inst.program_ref));
    head.appendChild(left);
    head.appendChild(el("span", "tag", inst.outcome));
    box.appendChild(head);

    var grid = el("div", "grid");

    // Doing now. storage_note MUST ride along here and not only on the no-table
    // branch: an instrument with rows is exactly the case where the note matters
    // ("this table is my INPUT, not my output"; "singleton -- no history exists").
    // Rendering it only when row_count is null dropped it for every instrument
    // that had one, which is the false-liveness reading the note exists to stop.
    if (inst.row_count !== null && inst.row_count !== undefined) {
      var rows = inst.row_count.toLocaleString() + " rows in " + inst.table;
      if (inst.storage_note) rows += " — " + inst.storage_note;
      grid.appendChild(cell("writing now", fmtAgo(inst.last_seen), rows));
    } else {
      grid.appendChild(cell("writing now", "no SQL table", inst.storage_note || inst.storage_kind));
    }

    // Historically -- and what bounds it. This is the cell that would have
    // stopped a 7-day window being read as "full history" in an Objective 7 pass.
    if (inst.history_hours !== null && inst.history_hours !== undefined) {
      var capped = inst.retention_hours && inst.history_hours >= inst.retention_hours * 0.9;
      var note = inst.retention_hours
        ? "capped by " + inst.retention_setting + "=" + inst.retention_hours +
          " (" + inst.retention_source + ")"
        : "no retention setting declared";
      grid.appendChild(cell("history", fmtHours(inst.history_hours), note, capped ? "capped" : null));
    } else {
      grid.appendChild(cell("history", "—", inst.storage_note || "no history table"));
    }

    // Affecting.
    if (inst.consumers && inst.consumers.length) {
      grid.appendChild(cell("affects", inst.consumers.length + " consumers",
        "resolved from the metric semantic layer"));
    } else {
      grid.appendChild(cell("affects", "—", inst.consumer_note || "not resolved (use the button above)"));
    }

    // Code.
    grid.appendChild(cell("code", broken ? "MISSING" : "present",
      inst.module + (inst.entrypoint ? "::" + inst.entrypoint : ""),
      broken ? "err" : null));

    box.appendChild(grid);

    if (inst.consumers && inst.consumers.length) {
      var det = el("details");
      det.appendChild(el("summary", null, "blast radius (" + inst.consumers.length + ")"));
      var ul = el("ul");
      inst.consumers.forEach(function (c) { ul.appendChild(el("li", null, c)); });
      det.appendChild(ul);
      box.appendChild(det);
    }

    var unlock = el("div", "unlock");
    var lab = el("span", "lab", "what this unlocks — reviewed " + inst.last_reviewed +
      (inst.review_stale ? " (STALE)" : ""));
    if (inst.review_stale) lab.className = "lab stale";
    unlock.appendChild(lab);
    unlock.appendChild(document.createTextNode(inst.unlock));
    box.appendChild(unlock);

    inst.claims.forEach(function (c) {
      var row = el("div", "claim");
      var line = el("div");
      line.appendChild(el("span", "cst " + c.status, c.status));
      line.appendChild(el("span", "q", c.question));
      if (c.blocks) line.appendChild(el("span", "blocks", "blocks " + c.blocks));
      row.appendChild(line);
      if (c.detail) row.appendChild(el("div", "cdet", c.detail));
      row.appendChild(el("div", "cnote",
        "recorded " + JSON.stringify(c.recorded) + " on " + c.recorded_at +
        (c.observed !== null && c.observed !== undefined ? " · live " + JSON.stringify(c.observed) : "")));
      if (c.note) row.appendChild(el("div", "cnote", c.note));
      box.appendChild(row);
    });

    return box;
  }

  function render() {
    renderOutcomes(state);
    var host = document.getElementById("instruments");
    host.textContent = "";
    state.instruments.forEach(function (i) { host.appendChild(renderInstrument(i)); });
  }

  function load(withConsumers) {
    var status = document.getElementById("status");
    status.className = "muted";
    status.textContent = withConsumers
      ? "resolving blast radius (scans ~4,300 files, takes a while)…"
      : "loading…";
    fetch("/api/sentience-program" + (withConsumers ? "?consumers=true" : ""))
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        state = data;
        render();
        var drift = 0, errs = 0;
        data.instruments.forEach(function (i) {
          i.claims.forEach(function (c) {
            if (c.status === "DRIFTED") drift += 1;
            if (c.status === "ERROR") errs += 1;
          });
        });
        // A database outage must read as an outage, never as an empty board.
        if (data.db_error) {
          status.className = "err";
          status.textContent = "database unavailable (" + data.db_error +
            ") — manifest and code state shown; live claims could not be checked";
          return;
        }
        status.textContent = data.instruments.length + " instruments · " +
          drift + " drifted · " + errs + " errored · read " + new Date().toLocaleTimeString();
        status.className = drift || errs ? "stale" : "muted";
      })
      .catch(function (e) {
        status.className = "err";
        status.textContent = "failed to load: " + e.message;
      });
  }

  document.getElementById("reload").addEventListener("click", function () { load(false); });
  document.getElementById("load-consumers").addEventListener("click", function () { load(true); });
  load(false);
})();

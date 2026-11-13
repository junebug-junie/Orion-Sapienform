/* Personality tab (templates/personality.html, scripts/personality_routes.py).
 *
 * Personality spec rev 4, move 4: the tab is built around Orion's CHOICES (what Orion picked, and
 * what Orion passed over), never around Orion's prose. Three parts:
 *   1. Recent choices      GET /api/personality/choices?limit=30
 *   2. Tendencies          GET /api/personality/tendencies?limit=20, each with a Yes / No:
 *                          POST /api/personality/tendencies/<id>/verdict {"verdict": "yes"|"no"}
 *                          A "no" retires the tendency (durable-runs writes it).
 *   3. Is it changing anything   GET /api/personality/arms?days=7
 * Pure view functions are exported for node:test. The DOM layer writes every piece of upstream text
 * with textContent (option texts and labels come from Orion's lanes), and never uses innerHTML.
 */
(function () {
  "use strict";

  const CHOICE_LIMIT = 30;
  const TENDENCY_LIMIT = 20;
  const ARM_DAYS = 7;
  const PASSED_OVER_SHOWN = 3;
  // orion/prime/tending.py MIN_WEIGHT: an active tendency at this weight reaches Prime's block.
  const PRIME_MIN_WEIGHT = 10;

  const LANE_TEXT = {
    prime: "Orion's own next move",
    outreach: "Reaching out to you",
    curiosity: "What to look into",
    reading: "What to read",
    chat: "In chat",
  };
  const DECLINE_TEXT = {
    prime: "Made no move",
    outreach: "Said nothing",
  };
  const DECLINE_TEXT_DEFAULT = "Picked none of them";
  const STATUS_TEXT = {
    active: "Orion leans on this now",
    candidate: "Still forming",
    retired: "Retired",
  };
  const ARM_TEXT = {
    care: "With Orion's tendencies counted",
    holdback: "Without them (the comparison group)",
  };

  function fmtAt(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function num(value) {
    const n = Number(value);
    return value == null || value === "" || !Number.isFinite(n) ? null : n;
  }

  function laneText(lane) {
    return LANE_TEXT[lane] || String(lane || "unknown lane");
  }

  function rankText(rank) {
    const r = num(rank);
    return r == null ? "not on the market's list" : `market's #${r}`;
  }

  function optionText(opt) {
    if (!opt || typeof opt !== "object") return "";
    const r = num(opt.market_rank);
    return r == null ? String(opt.text || "") : `#${r} ${String(opt.text || "")}`;
  }

  // One choice, in plain words: what was picked (or that Orion declined every option) and what
  // was passed over, ranked options first in the market's order.
  function choiceView(c) {
    const choice = c && typeof c === "object" ? c : {};
    const passed = Array.isArray(choice.passed_over) ? choice.passed_over : [];
    const declined = Boolean(choice.declined_all) || !choice.chosen;
    const picked = declined
      ? (DECLINE_TEXT[choice.lane] || DECLINE_TEXT_DEFAULT)
      : `Picked: ${String(choice.chosen.text || "")} (${rankText(choice.chosen.market_rank)})`;
    const shown = passed.slice(0, PASSED_OVER_SHOWN).map(optionText);
    const notes = [];
    if (!declined && num(choice.chosen.market_rank) === 1) notes.push("Same as the market's first pick.");
    if (choice.care_promoted) notes.push("Orion's care for a tendency moved this up.");
    const picks = num(choice.picks_in_decision);
    if (picks != null && picks > 1) notes.push(`One of ${picks} picks in this decision.`);
    const tendency = choice.tendency && choice.tendency.label ? `Counts toward: ${choice.tendency.label}` : "";
    return {
      id: String(choice.choice_id || ""),
      lane: laneText(choice.lane),
      when: fmtAt(choice.occurred_at),
      declined,
      picked,
      passedOverLabel: declined ? "Options Orion turned down" : "Passed over",
      passedOver: shown,
      allPassedOver: passed.map(optionText),
      more: Math.max(0, passed.length - shown.length),
      notes,
      tendency,
    };
  }

  // The reducer's label is "<chosen> over <declined>, n of m"; the count is shown on its own line.
  function labelBase(label) {
    return String(label || "").replace(/, \d+ of \d+$/, "") || "(no label)";
  }

  function verdictText(j) {
    if (!j || !j.verdict) return "You have not answered yet.";
    const when = fmtAt(j.at);
    const said = j.verdict === "yes" ? "yes" : "no";
    return when ? `You said ${said} on ${when}.` : `You said ${said}.`;
  }

  function exampleText(e) {
    const v = choiceView(e);
    const over = v.passedOver.length ? ` over ${v.passedOver.join(", ")}${v.more ? ` and ${v.more} more` : ""}` : "";
    const head = v.declined ? v.picked : v.picked.replace(/^Picked: /, "Picked ");
    return `${head}${over} (${v.lane}${v.when ? ", " + v.when : ""})`;
  }

  function tendencyView(t) {
    const tend = t && typeof t === "object" ? t : {};
    const weight = num(tend.weight);
    const n = num(tend.n);
    const m = num(tend.m);
    const status = String(tend.status || "");
    return {
      id: String(tend.disposition_id || ""),
      label: labelBase(tend.label),
      name: tend.name ? `Orion calls this: ${tend.name}` : "",
      status: STATUS_TEXT[status] || status || "unknown",
      count: n != null && m != null ? `${n} of ${m} choices went this way` : "",
      strength: weight == null ? "" : `Strength ${weight.toFixed(1)} (Prime sees it at ${PRIME_MIN_WEIGHT})`,
      inPrime: status === "active" && weight != null && weight >= PRIME_MIN_WEIGHT,
      examples: (Array.isArray(tend.examples) ? tend.examples : []).map(exampleText),
      verdict: verdictText(tend.juniper),
      answered: tend.juniper && tend.juniper.verdict ? String(tend.juniper.verdict) : "",
    };
  }

  function armsView(body) {
    const b = body && typeof body === "object" ? body : {};
    const days = num(b.days) || ARM_DAYS;
    const total = num(b.total) || 0;
    if (!total) {
      return {
        live: false,
        summary: `Not live yet. No market decision in the last ${days} days carries a care or holdback mark.`,
        rows: [],
      };
    }
    const arms = b.arms && typeof b.arms === "object" ? b.arms : {};
    const rows = Object.keys(arms).sort().map((arm) => {
      const outcomes = arms[arm] || {};
      const funded = num(outcomes.funded) || 0;
      const all = Object.values(outcomes).reduce((s, v) => s + (num(v) || 0), 0);
      const pct = all ? Math.round((100 * funded) / all) : 0;
      return { arm: ARM_TEXT[arm] || arm, text: `${all} decisions, ${funded} funded (${pct}%)` };
    });
    return { live: true, summary: `${total} market decisions in the last ${days} days.`, rows };
  }

  function choicesFrom(body) {
    if (Array.isArray(body)) return body;
    return body && Array.isArray(body.choices) ? body.choices : [];
  }

  function tendenciesFrom(body) {
    if (Array.isArray(body)) return body;
    return body && Array.isArray(body.tendencies) ? body.tendencies : [];
  }

  function verdictUrl(id) {
    return `/api/personality/tendencies/${encodeURIComponent(id)}/verdict`;
  }

  // What the card says after an answer is saved.
  function afterVerdictText(result) {
    if (result && result.retired) return "Retired. Orion will not lean on this any more.";
    return "Saved. You said yes.";
  }

  const api = {
    laneText, rankText, optionText, choiceView, labelBase, verdictText, exampleText, tendencyView, armsView,
    choicesFrom, tendenciesFrom, verdictUrl, afterVerdictText, PRIME_MIN_WEIGHT, LANE_TEXT,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  // --- DOM layer --------------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  async function readJson(resp) {
    let body = null;
    try { body = await resp.json(); } catch (_e) { body = null; }
    if (!resp.ok) throw new Error((body && body.detail) || `HTTP ${resp.status}`);
    return body;
  }

  async function getJson(url) {
    return readJson(await fetch(url, { headers: { Accept: "application/json" } }));
  }

  function renderChoices(list) {
    const host = $("pChoices");
    host.replaceChildren();
    if (!list.length) {
      host.appendChild(el("div", "muted", "No choices recorded yet."));
      return;
    }
    for (const c of list) {
      const v = choiceView(c);
      const card = el("div", "choice");
      const head = el("div", "row");
      head.appendChild(el("span", "badge", v.lane));
      head.appendChild(el("span", "muted", v.when));
      card.appendChild(head);
      card.appendChild(el("div", v.declined ? "picked declined" : "picked", v.picked));
      if (v.passedOver.length) {
        const line = el("div", "passed");
        line.appendChild(el("span", "muted", `${v.passedOverLabel}: `));
        line.appendChild(el("span", null, v.passedOver.join(" · ")));
        card.appendChild(line);
        if (v.more) {
          const det = el("details", "more");
          det.appendChild(el("summary", null, `and ${v.more} more`));
          const ul = el("ul");
          for (const t of v.allPassedOver.slice(v.passedOver.length)) ul.appendChild(el("li", null, t));
          det.appendChild(ul);
          card.appendChild(det);
        }
      }
      for (const n of v.notes) card.appendChild(el("div", "muted", n));
      if (v.tendency) card.appendChild(el("div", "muted", v.tendency));
      host.appendChild(card);
    }
  }

  async function sendVerdict(card, id, verdict) {
    const msg = card.querySelector(".verdict-msg");
    const buttons = card.querySelectorAll("button.verdict");
    if (verdict === "no"
        && !window.confirm("Retire this tendency? Orion stops leaning on it, and it cannot come back.")) {
      return;
    }
    buttons.forEach((b) => { b.disabled = true; });
    msg.classList.remove("err");
    msg.textContent = "Saving…";
    try {
      const body = await readJson(await fetch(verdictUrl(id), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ verdict }),
      }));
      msg.textContent = afterVerdictText(body);
      if (body && body.retired) {
        card.classList.add("retired");
        card.querySelector(".verdict-row").remove();
      } else {
        card.dataset.answered = verdict;
        buttons.forEach((b) => { b.disabled = false; });
      }
    } catch (err) {
      msg.textContent = `Not saved: ${err.message}`;
      msg.classList.add("err");
      buttons.forEach((b) => { b.disabled = false; });
    }
  }

  function renderTendencies(list) {
    const host = $("pTendencies");
    host.replaceChildren();
    if (!list.length) {
      host.appendChild(el("div", "muted",
        "No tendencies yet. One forms when Orion makes the same kind of choice a few times."));
      return;
    }
    for (const t of list) {
      const v = tendencyView(t);
      const card = el("div", "tendency");
      card.dataset.id = v.id;
      if (v.answered) card.dataset.answered = v.answered;
      card.appendChild(el("div", "tendency-label", v.label));
      if (v.name) card.appendChild(el("div", "muted", v.name));
      const facts = el("div", "row");
      facts.appendChild(el("span", "badge", v.status));
      if (v.inPrime) facts.appendChild(el("span", "badge prime", "In Orion's Prime block now"));
      if (v.count) facts.appendChild(el("span", "muted", v.count));
      if (v.strength) facts.appendChild(el("span", "muted", v.strength));
      card.appendChild(facts);
      if (v.examples.length) {
        card.appendChild(el("div", "muted", "Example choices:"));
        const ul = el("ul", "examples");
        for (const e of v.examples) ul.appendChild(el("li", null, e));
        card.appendChild(ul);
      }
      const row = el("div", "row verdict-row");
      row.appendChild(el("span", null, "Does this look like Orion?"));
      const yes = el("button", "verdict yes", "Yes");
      yes.type = "button";
      const no = el("button", "verdict no", "No");
      no.type = "button";
      yes.addEventListener("click", () => sendVerdict(card, v.id, "yes"));
      no.addEventListener("click", () => sendVerdict(card, v.id, "no"));
      row.appendChild(yes);
      row.appendChild(no);
      card.appendChild(row);
      card.appendChild(el("div", "muted verdict-msg", v.verdict));
      host.appendChild(card);
    }
  }

  function renderArms(body) {
    const v = armsView(body);
    $("pArmsSummary").textContent = v.summary;
    const host = $("pArms");
    host.replaceChildren();
    for (const r of v.rows) {
      const line = el("div", "arm");
      line.appendChild(el("strong", null, `${r.arm}: `));
      line.appendChild(el("span", null, r.text));
      host.appendChild(line);
    }
  }

  async function load() {
    const status = $("pStatus");
    status.textContent = "loading…";
    status.classList.remove("err");
    const [choices, tendencies, arms] = await Promise.allSettled([
      getJson(`/api/personality/choices?limit=${CHOICE_LIMIT}`),
      getJson(`/api/personality/tendencies?limit=${TENDENCY_LIMIT}`),
      getJson(`/api/personality/arms?days=${ARM_DAYS}`),
    ]);
    const errors = [];
    if (choices.status === "fulfilled") {
      renderChoices(choicesFrom(choices.value));
    } else {
      $("pChoices").replaceChildren();
      errors.push(`choices: ${choices.reason.message}`);
    }
    if (tendencies.status === "fulfilled") {
      renderTendencies(tendenciesFrom(tendencies.value));
      const warn = (tendencies.value && tendencies.value.warnings) || [];
      $("pTendencyWarn").textContent = warn.length ? `Answers cannot be saved yet: ${warn.join("; ")}` : "";
    } else {
      $("pTendencies").replaceChildren();
      $("pTendencyWarn").textContent = "";
      errors.push(`tendencies: ${tendencies.reason.message}`);
    }
    if (arms.status === "fulfilled") {
      renderArms(arms.value);
    } else {
      $("pArms").replaceChildren();
      $("pArmsSummary").textContent = "";
      errors.push(`market: ${arms.reason.message}`);
    }
    status.textContent = errors.length ? `Could not load: ${errors.join(" | ")}` : "";
    status.classList.toggle("err", errors.length > 0);
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("pRefresh").addEventListener("click", load);
    load();
  });
})();

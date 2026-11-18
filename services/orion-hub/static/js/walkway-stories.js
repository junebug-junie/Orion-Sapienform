(function (global) {
  // "Walkway stories" spot-check card, under the live picture in the Vision
  // panel's "Walkway (live)" view (motion stories slice 1, 2026-10-16;
  // docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md).
  //
  // orion-sql-writer turns each passing thing into one sentence from box
  // maths. Juniper keeps them HUB ONLY until she has checked them: >= 50
  // checked, >= 80% fully right, zero wrong names. This card lists recent
  // stories with the pictures they rest on and records Right / Wrong name /
  // Wrong motion / Wrong other (scripts/motion_story_routes.py).
  //
  // Pictures (2026-10-17): the picture of the MOMENT. orion-sql-writer saves,
  // while the frames still exist, up to 4 crops and one full frame with the
  // passage's boxes drawn (1 = first, green; last = red). Each picture is
  // labelled with its own time. A story without one says "No picture saved
  // for this story" and offers "Can't tell", which never counts as right or
  // wrong. "Only checkable" (on by default) lists stories with a saved frame
  // that is not dark, so the queue is all judgeable. The old box map drew the
  // boxes over today's view, which at night "just looks dark" -- removed.
  //
  // Who the story is about (2026-10-18): one box per person in the story (its
  // walker, and anyone it walked with) shows the name Orion has for them, or
  // "unnamed person (individual <short id>)". Juniper can type a name (it is a
  // "who is this?" answer: the same recognition every other card feeds), say
  // "Someone else" (optionally naming who it really is: those pictures are
  // split off that individual), or "Not sure". The sentence shown is the
  // server's display_sentence (her answer on this story wins; otherwise the
  // strict naming rule on today's names); if it differs from what Orion first
  // wrote, that is shown too.
  //
  // Everything from the database is rendered with textContent / attributes,
  // never innerHTML.

  const VIEW_VALUE = "walkway-live";
  const POLL_MS = 60000;
  const VERDICTS = [
    { value: "right", label: "Right" },
    { value: "wrong_name", label: "Wrong name" },
    { value: "wrong_motion", label: "Wrong motion" },
    { value: "wrong_other", label: "Wrong other" },
  ];
  // Fifth button: Juniper could not judge it. Never right, never wrong.
  const CANT_TELL = { value: "cant_tell", label: "Can't tell" };
  const THUMB_RE = /^[0-9a-f]{64}$/;
  const STORY_RE = /^[0-9a-f]{32}$/;
  const STRIP_FILE_RE = /^(crop_[0-3]|frame)\.jpg$/;
  const MISSING_TEXT = {
    frame_gone: "frame already gone",
    prop_overlap: "on the patio prop, not saved",
    empty_box: "box too small",
    not_captured: "not saved",
  };

  function tallyLine(tally) {
    const t = tally || {};
    const gate = t.gate || { min_checked: 50, min_right: 0.8, max_wrong_names: 0 };
    const checked = Number(t.checked || 0);
    const pct = typeof t.pct_right === "number" ? `${Math.round(t.pct_right * 100)}%` : "–";
    return (
      `Checked ${checked} of ${gate.min_checked} · right ${t.right || 0} (${pct}, need ` +
      `${Math.round(gate.min_right * 100)}%) · wrong names ${t.wrong_name || 0} (must be ${gate.max_wrong_names}) · ` +
      `named stories checked ${t.named_checked || 0} · can't tell ${t.cant_tell || 0} (not counted)`
    );
  }

  function gateLine(tally) {
    const t = tally || {};
    const gate = t.gate || { min_checked: 50, min_right: 0.8, max_wrong_names: 0 };
    const skipped = Number(t.not_checkable || 0);
    const scope = skipped
      ? ` The gate covers daylight stories with a picture; ${skipped} dark or picture-less ${skipped === 1 ? "story is" : "stories are"} not in it.`
      : "";
    if (gate.passed) return `Gate passed: stories may be switched on for Orion (Juniper decides).${scope}`;
    const reasons = [];
    const checked = Number(t.checked || 0);
    if (checked < gate.min_checked) reasons.push(`${gate.min_checked - checked} more to check`);
    if (checked && typeof t.pct_right === "number" && t.pct_right < gate.min_right) reasons.push("too many wrong");
    if ((t.wrong_name || 0) > gate.max_wrong_names) reasons.push("a wrong name was found");
    return `Hub only for now: ${reasons.join(", ") || "not yet"}.${scope}`;
  }

  // First, middle and last thumbnails that exist, in time order.
  function pickThumbs(evidence, max) {
    const n = max || 3;
    const have = (evidence || []).filter((e) => e && typeof e.thumb === "string" && THUMB_RE.test(e.thumb));
    if (have.length <= n) return have.map((e) => e.thumb);
    const idx = [0, Math.floor((have.length - 1) / 2), have.length - 1];
    return idx.slice(0, n).map((i) => have[i].thumb);
  }

  function stripUrl(storyId, file) {
    return `/api/vision/motion-stories/${storyId}/strip/${file}`;
  }

  // What the card can show for the moment itself. Only names the server
  // could have written are turned into URLs.
  function stripModel(story) {
    const st = story && story.evidence_strip;
    const id = story && STORY_RE.test(String(story.story_id || "")) ? story.story_id : null;
    const out = { frame: null, crops: [], dark: !!(story && story.dark), pruned: false, saved: false };
    if (!st || typeof st !== "object" || !id) return out;
    out.saved = true;
    out.pruned = !!st.pruned_at;
    out.retentionDays = typeof st.retention_days === "number" && Number.isFinite(st.retention_days)
      ? st.retention_days : null;
    if (out.pruned) return out;
    const f = st.frame;
    if (f && typeof f.file === "string" && STRIP_FILE_RE.test(f.file)) {
      out.frame = { url: stripUrl(id, f.file), t: f.t };
    }
    (Array.isArray(st.crops) ? st.crops : []).forEach((c) => {
      if (!c || typeof c !== "object") return;
      if (typeof c.file === "string" && STRIP_FILE_RE.test(c.file)) out.crops.push({ url: stripUrl(id, c.file), t: c.t });
      else out.crops.push({ missing: MISSING_TEXT[c.missing] ? c.missing : "not_captured", t: c.t });
    });
    return out;
  }

  function hasPicture(model) {
    return !!(model.frame || model.crops.some((c) => c.url));
  }

  function shortTime(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "?";
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function localTime(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso || "");
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function el(doc, tag, cls, text) {
    const n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  // How one person in a story is called on the card.
  function personTitle(p) {
    const kind = p && p.kind ? String(p.kind) : "person";
    const who = p && p.label ? String(p.label) : `unnamed ${kind} (individual ${String((p && p.short_id) || "").slice(0, 8)})`;
    return p && p.role === "companion" ? `With them: ${who}` : who;
  }

  function actionLine(a) {
    if (!a || !a.action) return "";
    if (a.action === "name") return `You named them ${a.name || ""} on this story.`;
    if (a.action === "someone_else") return a.name
      ? `You said this is someone else: ${a.name}.` : "You said this is someone else.";
    if (a.action === "unsure") return "You said you are not sure.";
    return "";
  }

  function renderPerson(doc, story, p, onPerson) {
    const box = el(doc, "div", "flex flex-wrap items-center gap-2 rounded border border-gray-700 bg-gray-900/60 p-2");
    box.setAttribute("data-story-person", p.individual_id);
    if (p.thumb && THUMB_RE.test(p.thumb)) {
      const img = el(doc, "img", "h-12 w-auto rounded border border-gray-700");
      img.setAttribute("src", `/api/vision/crop-thumbs/${p.thumb}`);
      img.setAttribute("alt", "this person in the story");
      img.setAttribute("loading", "lazy");
      img.addEventListener("error", () => img.remove());
      box.appendChild(img);
    }
    const text = el(doc, "div", "min-w-0 flex-1 space-y-0.5");
    const title = el(doc, "div", "text-xs text-gray-100", personTitle(p));
    title.setAttribute("data-person-title", "1");
    text.appendChild(title);
    if (p.label && p.label_copied) {
      text.appendChild(el(doc, "div", "text-[10px] text-gray-500", "Name carried over from a same-person answer."));
    }
    const said = actionLine(p.story_action);
    if (said) {
      const line = el(doc, "div", "text-[10px] text-emerald-300", said);
      line.setAttribute("data-person-said", p.story_action.action);
      text.appendChild(line);
    }
    box.appendChild(text);
    const input = el(doc, "input", "min-h-[2.25rem] w-full sm:w-40 rounded border border-gray-600 bg-gray-950 px-2 text-xs text-gray-100");
    input.setAttribute("type", "text");
    input.setAttribute("maxlength", "120");
    input.setAttribute("placeholder", p.label ? "Correct the name" : "Who is this?");
    input.setAttribute("aria-label", `Name for ${personTitle(p)}`);
    input.setAttribute("data-person-name-input", "1");
    box.appendChild(input);
    const buttons = el(doc, "div", "flex flex-wrap gap-2");
    [{ value: "name", label: p.label ? "Save name" : "Name them" },
      { value: "someone_else", label: "Someone else" },
      { value: "unsure", label: "Not sure" }].forEach((a) => {
      const btn = el(doc, "button", "min-h-[2.25rem] rounded-lg border border-gray-600 bg-gray-900 px-3 py-1 text-xs text-gray-200 hover:bg-gray-700", a.label);
      btn.setAttribute("type", "button");
      btn.setAttribute("data-person-action", a.value);
      if (a.value === "someone_else") btn.setAttribute("title", "Not this individual. Type who it really is first, if you know.");
      btn.addEventListener("click", () => onPerson(story, p, a.value, input.value, box));
      buttons.appendChild(btn);
    });
    box.appendChild(buttons);
    return box;
  }

  // Street continuity (2026-10-20): the other camera's sighting of this passage,
  // from the handoff reducer. {kind: "from" | "next" | "same", camera, seconds}.
  function handoffText(h) {
    if (!h || !h.camera) return "";
    const secs = typeof h.seconds === "number" ? h.seconds : null;
    if (h.kind === "same") return `The ${h.camera} saw them at the same moment.`;
    if (secs === null) return "";
    if (h.kind === "from") return `Came from the ${h.camera} ${secs} s earlier.`;
    if (h.kind === "next") return `Seen next on the ${h.camera} ${secs} s later.`;
    return "";
  }

  function renderStory(doc, story, onVerdict, onPerson) {
    const item = el(doc, "div", "rounded-lg border border-gray-700 bg-gray-800/60 p-3 space-y-2");
    item.setAttribute("data-story-id", story.story_id);
    const shown = story.display_sentence || story.sentence;
    const sentence = el(doc, "div", "text-sm text-gray-100", shown);
    sentence.setAttribute("data-story-sentence", "1");
    item.appendChild(sentence);
    if (story.display_sentence && story.display_sentence !== story.sentence) {
      const was = el(doc, "div", "text-[10px] text-gray-500", `Orion first wrote: ${story.sentence}`);
      was.setAttribute("data-story-first-written", "1");
      item.appendChild(was);
    }
    const handoff = handoffText(story.handoff);
    if (handoff) {
      const line = el(doc, "div", "text-[11px] text-sky-300", handoff);
      line.setAttribute("data-story-handoff", story.handoff.kind);
      item.appendChild(line);
    }
    const meta = [localTime(story.started_at), `${story.kind}/${story.speed_class || "?"}`,
      `${story.detection_count} detections`];
    if (story.name_source === "juniper") meta.push("named by you");
    else if (story.display_name !== undefined ? story.display_name : story.name_label) meta.push("named");
    item.appendChild(el(doc, "div", "text-[11px] text-gray-400", meta.join(" · ")));

    const model = stripModel(story);
    const pics = el(doc, "div", "space-y-1");
    pics.setAttribute("data-story-strip", model.frame ? "frame" : hasPicture(model) ? "crops" : "none");
    if (model.dark) pics.appendChild(el(doc, "div", "text-[11px] font-semibold text-amber-300", "Dark scene"));
    if (model.frame) {
      const img = el(doc, "img", "w-full max-w-[20rem] h-auto rounded border border-gray-700");
      img.setAttribute("src", model.frame.url);
      img.setAttribute("alt", "the walkway at the moment of this story, boxes drawn");
      img.setAttribute("loading", "lazy");
      img.setAttribute("data-story-frame", "1");
      img.addEventListener("error", () => img.replaceWith(el(doc, "span", "text-[10px] text-gray-500", "(picture gone)")));
      pics.appendChild(img);
      pics.appendChild(el(doc, "div", "text-[10px] text-gray-500",
        `Picture from ${localTime(model.frame.t)}. Boxes: 1 = first (green), last = red.`));
    }
    if (model.crops.length) {
      const row = el(doc, "div", "flex flex-wrap items-start gap-2");
      model.crops.forEach((c) => {
        const cell = el(doc, "div", "flex flex-col items-center");
        if (c.url) {
          const img = el(doc, "img", "h-24 w-auto max-w-[5rem] object-contain rounded border border-gray-700");
          img.setAttribute("src", c.url);
          img.setAttribute("alt", `crop from ${shortTime(c.t)}`);
          img.setAttribute("loading", "lazy");
          img.setAttribute("data-story-crop", "1");
          img.addEventListener("error", () => img.replaceWith(el(doc, "span", "text-[10px] text-gray-500", "(picture gone)")));
          cell.appendChild(img);
          cell.appendChild(el(doc, "span", "text-[10px] text-gray-500", shortTime(c.t)));
        } else {
          const why = MISSING_TEXT[c.missing] || MISSING_TEXT.not_captured;
          cell.appendChild(el(doc, "span", "text-[10px] text-gray-500", `${shortTime(c.t)}: ${why}`));
          cell.setAttribute("data-story-crop-missing", c.missing);
        }
        row.appendChild(cell);
      });
      pics.appendChild(row);
    }
    if (!hasPicture(model)) {
      // vision-host's own thumbnails (a few boxes) are still pictures of the moment.
      const thumbs = pickThumbs(story.evidence);
      const row = el(doc, "div", "flex flex-wrap items-start gap-2");
      thumbs.forEach((h) => {
        const img = el(doc, "img", "h-20 w-auto rounded border border-gray-700");
        img.setAttribute("src", `/api/vision/crop-thumbs/${h}`);
        img.setAttribute("alt", "crop from this story");
        img.setAttribute("loading", "lazy");
        img.setAttribute("data-story-thumb", h);
        img.addEventListener("error", () => img.replaceWith(el(doc, "span", "text-[10px] text-gray-500", "(picture gone)")));
        row.appendChild(img);
      });
      if (thumbs.length) pics.appendChild(row);
      pics.appendChild(el(doc, "div", "text-[11px] text-gray-400",
        model.pruned
          ? `Picture deleted ${model.retentionDays != null ? `${model.retentionDays} days` : "some time"} after the story (it already has a verdict).`
          : thumbs.length ? "No full picture saved for this story; only these small crops."
            : "No picture saved for this story."));
    }
    item.appendChild(pics);

    const people = Array.isArray(story.people) ? story.people.filter((p) => p && p.individual_id) : [];
    const who = el(doc, "div", "space-y-1");
    who.setAttribute("data-story-people", String(people.length));
    who.appendChild(el(doc, "div", "text-[11px] font-semibold text-gray-300",
      people.length ? "Who is this story about?" : "No individual linked to these boxes yet."));
    if (typeof onPerson === "function") people.forEach((p) => who.appendChild(renderPerson(doc, story, p, onPerson)));
    item.appendChild(who);

    const row = el(doc, "div", "flex flex-wrap gap-2");
    // "Can't tell" on every story: an unclear daylight picture must not be forced into "wrong".
    const buttons = VERDICTS.concat([CANT_TELL]);
    buttons.forEach((v) => {
      const btn = el(doc, "button", "", v.label);
      btn.setAttribute("type", "button");
      btn.setAttribute("data-story-verdict", v.value);
      setPressed(btn, story.verdict === v.value);
      btn.addEventListener("click", () => onVerdict(story, v.value, item));
      row.appendChild(btn);
    });
    item.appendChild(row);
    return item;
  }

  function setPressed(btn, on) {
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.className = "min-h-[2.25rem] rounded-lg border px-3 py-1 text-xs " + (on
      ? "border-emerald-400 bg-emerald-700 text-white"
      : "border-gray-600 bg-gray-900 text-gray-200 hover:bg-gray-700");
  }

  function mount(doc, fetchFn) {
    const card = doc.getElementById("walkwayStoriesCard");
    const select = doc.getElementById("visionSource");
    if (!card) return null;
    const list = doc.getElementById("walkwayStoriesList");
    const status = doc.getElementById("walkwayStoriesStatus");
    const tallyEl = doc.getElementById("walkwayStoriesTally");
    const gateEl = doc.getElementById("walkwayStoriesGate");
    const uncheckedBox = doc.getElementById("walkwayStoriesUnchecked");
    const checkableBox = doc.getElementById("walkwayStoriesCheckable");
    let timer = null;

    function showTally(t) {
      if (tallyEl) tallyEl.textContent = tallyLine(t);
      if (gateEl) gateEl.textContent = gateLine(t);
    }

    async function onVerdict(story, verdict, item) {
      const buttons = item.querySelectorAll("[data-story-verdict]");
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetchFn(`/api/vision/motion-stories/${encodeURIComponent(story.story_id)}/verdict`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ verdict }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        story.verdict = verdict;
        buttons.forEach((b) => setPressed(b, b.getAttribute("data-story-verdict") === verdict));
        showTally(body.tally);
        if (status) status.textContent = "Saved.";
      } catch (err) {
        if (status) status.textContent = `Could not save that verdict (${err.message}). Try again.`;
      } finally {
        buttons.forEach((b) => { b.disabled = false; });
      }
    }

    async function onPerson(story, person, action, typed, box) {
      const name = String(typed || "").trim().replace(/\s+/g, " ");
      if (action === "name" && !name) {
        if (status) status.textContent = "Type a name first.";
        return;
      }
      const buttons = box.querySelectorAll("[data-person-action]");
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetchFn(
          `/api/vision/motion-stories/${encodeURIComponent(story.story_id)}/people/${encodeURIComponent(person.individual_id)}`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action, name: action === "unsure" ? "" : name }),
          });
        if (!res.ok) {
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (_) { /* not JSON */ }
          throw new Error(detail ? `${res.status} ${detail}` : `HTTP ${res.status}`);
        }
        const body = await res.json();
        if (body.tally) showTally(body.tally);
        const item = box.closest("[data-story-id]");
        if (body.story && item) item.replaceWith(renderStory(doc, body.story, onVerdict, onPerson));
        if (status) {
          status.textContent = action === "name"
            ? `Saved: ${name}. Orion applies it like any "who is this?" answer.`
            : action === "someone_else" ? "Saved: someone else. Orion will try to split those pictures off that individual."
              : "Saved: not sure.";
          if (body.wrong_name_recorded) status.textContent += " Recorded as a wrong name for this story.";
        }
      } catch (err) {
        if (status) status.textContent = `Could not save that (${err.message}). Try again.`;
        buttons.forEach((b) => { b.disabled = false; });
      }
    }

    async function load(fromTimer) {
      const unchecked = uncheckedBox && uncheckedBox.checked ? "true" : "false";
      const checkable = !checkableBox || checkableBox.checked ? "true" : "false";
      try {
        const res = await fetchFn(
          `/api/vision/motion-stories?limit=20&unchecked_only=${unchecked}&checkable_only=${checkable}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        showTally(body.tally);
        // The 60 s refresh never wipes a name Juniper is typing (a filter change she makes does reload).
        const typing = fromTimer === true && Array.from(list.querySelectorAll("[data-person-name-input]"))
          .some((i) => i.value.trim() || doc.activeElement === i);
        if (typing) return;
        list.textContent = "";
        const stories = body.stories || [];
        stories.forEach((s) => list.appendChild(renderStory(doc, s, onVerdict, onPerson)));
        if (status) {
          const which = checkable === "true" ? "checkable: daylight, with a picture" : "all, including dark or no picture";
          status.textContent = stories.length
            ? `${stories.length} recent ${stories.length === 1 ? "story" : "stories"} (${which}).`
            : checkable === "true"
              ? "No checkable stories yet. Untick \"only checkable\" to see dark ones and ones without a picture."
              : "No stories yet. Someone has to walk past first (walkers only: runners, bikes and cars are too fast for the camera today).";
        }
      } catch (err) {
        if (status) status.textContent = `Can't load walkway stories (${err.message}).`;
      }
    }

    function sync() {
      const on = !!select && select.value === VIEW_VALUE;
      card.classList.toggle("hidden", !on);
      if (on && !timer) {
        load();
        timer = setInterval(() => load(true), POLL_MS);
      } else if (!on && timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    if (select) select.addEventListener("change", sync);
    if (uncheckedBox) uncheckedBox.addEventListener("change", () => load(false));
    if (checkableBox) checkableBox.addEventListener("change", () => load(false));
    sync();
    return { load, sync };
  }

  const api = { VIEW_VALUE, VERDICTS, CANT_TELL, tallyLine, gateLine, pickThumbs, stripModel, hasPicture,
                personTitle, actionLine, handoffText, renderStory, mount };

  global.OrionWalkwayStories = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (typeof document !== "undefined" && typeof global.fetch === "function") {
    const start = function () {
      api.mount(document, global.fetch.bind(global));
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }
})(typeof window !== "undefined" ? window : globalThis);

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
  // One person at a time (2026-11-24, Juniper: "how the f would i know which is
  // which, [name] or [name]?"). Each person gets a letter (A, B, ...). Their row
  // shows THEIR OWN pictures (the strip crops that are theirs, by crop_id; else
  // vision-host's thumbnail of one of their crops), and says which numbered
  // boxes on the big picture are theirs, in the colour that box was drawn. The
  // time crops above carry the same letter. A row with a name gets "Yes, that's
  // <name>" (a "same" picture verdict). A person with no picture of their own is
  // greyed: "No picture of this person - not asked." (the route refuses it too).
  //
  // How she could tell (2026-11-24): an optional "How could you tell?" box per
  // person goes with whichever answer she gives ("pushing a stroller; Jason has
  // no young kids; not his build"). Stored as the answer's `why`, shown back
  // under what she said. Raw context clues for a later primitive to read.
  //
  // One person per linked group (2026-11-24, who-card redesign Patch A). Camera
  // groups Juniper already said are the same person (a "same" link, mostly made
  // when she typed a name she used before) are ONE row: their pictures together,
  // "Seen as N camera groups in this story", asked once. The answer is written for
  // each camera group (`group: true`). A wrong link would hide here, so a grouped
  // row also offers "These are N camera groups - not the same person?", which
  // records her "same person? no" for one direct link (scripts/motion_story_routes.py).
  //
  // One person at a time (2026-11-24, who-card redesign Patch B). The card asks
  // about ONE person: their pictures, "Is this Sasha?" (or "Who is this?"), and
  // three answers: "Yes, that's Sasha", "No, it's ..." and "Can't see them".
  // "No, it's ..." opens a short list of the names Orion knows (most seen first),
  // "Someone not listed" (type a name) and "I don't know who". A name picked from
  // the list is one tap with no confirm step, so it is held as "Changed to X - Undo"
  // and only sent when she moves to the next person (or leaves the page). Then
  // "Next person (2 of 4)". A person with no picture of their own is never asked.
  // Answers carry `card: "one_at_a_time"`: a "not sure" story the server put back
  // in the unchecked queue leaves it on her first answer here.
  //
  // The sentence graded in parts (2026-11-24, who-card redesign Patch C). Under
  // the sentence: "Right people?", "Right direction?", "Right place?" -- only the
  // parts this sentence states (recorded when it was written, never guessed from
  // its words) -- each Yes / No / Can't tell. The story's verdict is made from
  // them once every part is answered (people no -> wrong name first). "Yes, that's
  // X" on a person never answers "Right people?" (Juniper's answer 2). The tally
  // line says how many verdicts came from parts and how many from the old buttons.
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
  const VERDICT_WORDS = { right: "right", wrong_name: "a wrong name", wrong_motion: "a wrong motion",
    wrong_other: "wrong (other)", cant_tell: "can't tell" };
  const PART_QUESTIONS = { people: "Right people?", direction: "Right direction?", place: "Right place?" };
  const PART_ANSWERS = [{ value: "yes", label: "Yes" }, { value: "no", label: "No" },
    { value: "cant_tell", label: "Can't tell" }];
  const WHY_MAX_CHARS = 1000;
  const CARD = "one_at_a_time";
  const NAMES_LIMIT = 8;
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
      `named stories checked ${t.named_checked || 0} · can't tell ${t.cant_tell || 0} (not counted)` +
      // "Can't see them" answers apart: a rising share means the pictures are too small or dark.
      (typeof t.person_answers === "number"
        ? ` · can't see them ${t.cant_see || 0} of ${t.person_answers} person answers` : "") +
      sourceText(t)
    );
  }

  // Where the counted verdicts came from (Orion's question 5): the old buttons, the sentence
  // parts, a name correction; and stories graded in part only, which count for nothing yet.
  function sourceText(t) {
    const by = t && t.by_source;
    if (!by || typeof by !== "object") return "";
    const n = (src) => Object.keys(by[src] || {}).filter((k) => k !== "cant_tell")
      .reduce((a, k) => a + Number(by[src][k] || 0), 0);
    let out = ` · from parts ${n("parts")}, from the old buttons ${n("button")}`;
    if (n("name_correction")) out += `, from name corrections ${n("name_correction")}`;
    if (Number(t.wrong_name_from_corrections || 0)) {
      out += ` · wrong names held by your corrections ${t.wrong_name_from_corrections}`;
    }
    if (Number(t.half_graded || 0)) out += ` · half-graded ${t.half_graded} (not counted yet)`;
    return out;
  }

  // The parts to grade for a story: the recorded ones, in order; null when not recorded yet.
  function storyParts(story) {
    const stated = story && story.parts && Array.isArray(story.parts.stated) ? story.parts.stated : null;
    return stated ? stated.filter((p) => Object.prototype.hasOwnProperty.call(PART_QUESTIONS, p)) : null;
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
      if (typeof c.file === "string" && STRIP_FILE_RE.test(c.file)) {
        out.crops.push({ url: stripUrl(id, c.file), t: c.t, crop_id: typeof c.crop_id === "string" ? c.crop_id : null });
      }
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
    if (a.action === "confirm") return `You said yes, this is ${a.name || "them"}.`;
    if (a.action === "someone_else") return a.name
      ? `You said this is someone else: ${a.name}.` : "You said this is someone else.";
    if (a.action === "unsure") return a.dont_know ? "You said you don't know who this is." : "You said you can't see them.";
    return "";
  }

  const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  // A chip colour per person letter; never the box colours (green / blue / red).
  const CHIP = ["bg-amber-500 text-black", "bg-fuchsia-500 text-black", "bg-cyan-400 text-black",
    "bg-lime-400 text-black", "bg-orange-400 text-black", "bg-pink-400 text-black"];
  // The colour sql-writer drew each numbered box in (app/vision_story_strip.py).
  const BOX_TEXT = { green: "text-green-400", blue: "text-blue-400", red: "text-red-400" };

  // The story's persons, each with a letter: the server's linked groups (`persons`),
  // else one per camera group (`people`, an older server).
  function storyPeople(story) {
    const seen = new Set();
    const out = [];
    const persons = Array.isArray(story && story.persons) ? story.persons : null;
    (persons || (Array.isArray(story && story.people) ? story.people : [])).forEach((raw) => {
      if (!raw) return;
      const p = persons ? Object.assign({}, raw, { individual_id: raw.person_key, grouped: true }) : raw;
      if (!p.individual_id || seen.has(p.individual_id)) return;
      seen.add(p.individual_id);
      out.push(p);
    });
    return out.map((p, i) => ({
      person: p,
      letter: i < LETTERS.length ? LETTERS[i] : LETTERS[i % LETTERS.length] + Math.floor(i / LETTERS.length),
      chip: CHIP[i % CHIP.length],
    }));
  }

  // The pictures a person row shows: their own strip crops, else their thumbnail.
  function personPictures(p) {
    const out = [];
    (Array.isArray(p && p.pictures) ? p.pictures : []).forEach((c) => {
      if (c && STORY_RE.test(String(c.story_id || "")) && typeof c.file === "string" && STRIP_FILE_RE.test(c.file)) {
        out.push({ url: stripUrl(c.story_id, c.file), t: c.t, crop_id: c.crop_id });
      } else if (c && typeof c.thumb === "string" && THUMB_RE.test(c.thumb)) {
        // A camera group of a linked person that has only vision-host's thumbnail.
        out.push({ url: `/api/vision/crop-thumbs/${c.thumb}`, t: null, crop_id: c.crop_id || null });
      }
    });
    if (!out.length && p && typeof p.thumb === "string" && THUMB_RE.test(p.thumb)) {
      out.push({ url: `/api/vision/crop-thumbs/${p.thumb}`, t: null, crop_id: p.thumb_crop_id || null });
    }
    return out;
  }

  // Is this person askable? Only when the server says it has a picture of them
  // (the route refuses the rest) and the card can show one.
  function isPictured(p) {
    return !!p && p.pictured === true && personPictures(p).length > 0;
  }

  // crop_id -> {letter, chip} for the story's time crops.
  function cropOwners(entries) {
    const out = {};
    entries.forEach((e) => {
      (e.person.crop_ids || []).forEach((cid) => { if (!(cid in out)) out[cid] = e; });
    });
    return out;
  }

  // "Boxes 2, 3 on the big picture", "Not boxed on the big picture (they walked alongside)".
  function boxesText(p, hasFrame) {
    const boxes = Array.isArray(p && p.boxes) ? p.boxes.filter((b) => b && Number.isFinite(Number(b.n))) : [];
    if (!hasFrame) return "";
    if (!boxes.length) {
      return p && p.role === "companion" ? "Not boxed on the big picture (they walked alongside)."
        : "Not boxed on the big picture.";
    }
    return `${boxes.length === 1 ? "Box" : "Boxes"} ${boxes.map((b) => b.n).join(", ")} on the big picture.`;
  }

  // "Seen as 2 camera groups in this story." for a linked person; "" for one group.
  function groupsText(p) {
    const n = Array.isArray(p && p.individual_ids) ? p.individual_ids.length : 1;
    return n > 1 ? `Seen as ${n} camera groups in this story (you said they are the same person).` : "";
  }

  // The "not the same person?" choices: one per direct link between two of its groups.
  function notSameChoices(p) {
    const groups = Array.isArray(p && p.groups) ? p.groups : [];
    const letter = {};
    groups.forEach((g, i) => { letter[g.individual_id] = `group ${String(g.short_id || g.individual_id || "").slice(0, 8)}`; });
    return (Array.isArray(p && p.links) ? p.links : [])
      .filter((l) => l && letter[l.a] && letter[l.b])
      .map((l) => ({ a: l.a, b: l.b, label: `${letter[l.a]} and ${letter[l.b]} are different people` }));
  }

  function chipEl(doc, entry) {
    const chip = el(doc, "span", `inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded px-1 text-[11px] font-bold ${entry.chip}`, entry.letter);
    chip.setAttribute("data-person-letter", entry.letter);
    return chip;
  }

  // Which asked person a story shows: the one the cursor names; else the first she has not
  // answered on this story; else the first. A person who left the list (no picture after an
  // answer) never shifts the card onto someone else by index.
  function cursorIndex(askable, key) {
    const at = key ? askable.findIndex((e) => e.person.individual_id === key) : -1;
    if (at >= 0) return at;
    const open = askable.findIndex((e) => !e.person.story_action);
    return open >= 0 ? open : 0;
  }

  // The question for one person.
  function questionText(p) {
    const lead = p && p.role === "companion" ? "Walking with them: " : "";
    return p && p.label ? `${lead}Is this ${p.label}?` : `${lead}Who is this?`;
  }

  // The names offered under "No, it's ...": Orion's known names, most seen first, never
  // the name this person already has (that would be "yes"), only for people.
  function nameChoices(p, names) {
    if (p && p.kind && p.kind !== "person") return [];
    const own = String((p && p.label) || "").trim().toLowerCase();
    const seen = new Set();
    const out = [];
    (Array.isArray(names) ? names : []).forEach((n) => {
      const name = String((n && n.name) || "").trim();
      const key = name.toLowerCase();
      if (!name || key === own || seen.has(key)) return;
      seen.add(key);
      out.push(name);
    });
    return out.slice(0, NAMES_LIMIT);
  }

  // What one answer sends, from what she pressed. A picked or typed name on a named person
  // is "someone else, called X" (those pictures are not them); on an unnamed person it names them.
  function answerSpec(p, choice, name) {
    const named = !!(p && p.label);
    if (choice === "yes") return { action: "confirm", name: (p && p.label) || "" };
    if (choice === "cant_see") return { action: "unsure", name: "" };
    if (choice === "dont_know") return named ? { action: "someone_else", name: "" } : { action: "unsure", name: "", dontKnow: true };
    return { action: named ? "someone_else" : "name", name: String(name || "").trim().replace(/\s+/g, " ") };
  }

  function pendingText(p, spec) {
    return p && p.label ? `Changed to ${spec.name} (not ${p.label})` : `Named ${spec.name}`;
  }

  function btnEl(doc, label, attr, value, primary) {
    const btn = el(doc, "button", "min-h-[2.25rem] rounded-lg border px-3 py-1 text-xs " + (primary
      ? "border-emerald-500 bg-emerald-800 text-white hover:bg-emerald-700"
      : "border-gray-600 bg-gray-900 text-gray-200 hover:bg-gray-700"), label);
    btn.setAttribute("type", "button");
    btn.setAttribute(attr, value);
    return btn;
  }

  // One person, asked. ``ctx``: {answer(story, p, spec, box, held), undo(story), pending, names, onNotSame}.
  function renderAsk(doc, story, entry, hasFrame, ctx) {
    const p = entry.person;
    const pics = personPictures(p);
    const box = el(doc, "div", "space-y-2 rounded border border-gray-700 bg-gray-900/60 p-2");
    box.setAttribute("data-story-person", p.individual_id);
    box.setAttribute("data-person-letter", entry.letter);
    box.setAttribute("data-person-asked", "1");

    const head = el(doc, "div", "flex flex-wrap items-center gap-2");
    head.appendChild(chipEl(doc, entry));
    const q = el(doc, "div", "text-sm font-semibold text-gray-100", questionText(p));
    q.setAttribute("data-person-question", "1");
    q.setAttribute("tabindex", "-1");
    head.appendChild(q);
    const title = el(doc, "span", "sr-only", personTitle(p));
    title.setAttribute("data-person-title", "1");
    head.appendChild(title);
    if (p.label) {
      // Two groups can carry one name (two "Sasha"s): the group id tells them apart.
      head.appendChild(el(doc, "span", "text-[10px] text-gray-500", `group ${String(p.short_id || "").slice(0, 8)}`));
    }
    box.appendChild(head);

    // Their pictures: the first (largest box) big, the rest small.
    const row = el(doc, "div", "flex flex-wrap items-end gap-2");
    pics.forEach((c, i) => {
      const cell = el(doc, "div", "flex flex-col items-center");
      const img = el(doc, "img", (i === 0 ? "h-36 max-w-[8rem]" : "h-20 max-w-[4rem]") +
        " w-auto object-contain rounded border-2 border-gray-500");
      img.setAttribute("src", c.url);
      img.setAttribute("alt", `Person ${entry.letter} (${personTitle(p)})${c.t ? ` at ${shortTime(c.t)}` : ""}`);
      img.setAttribute("loading", "lazy");
      img.setAttribute("data-person-picture", c.crop_id || "1");
      img.addEventListener("error", () => img.replaceWith(el(doc, "span", "text-[10px] text-gray-500", "(picture gone)")));
      cell.appendChild(img);
      if (c.t) cell.appendChild(el(doc, "span", "text-[10px] text-gray-500", shortTime(c.t)));
      row.appendChild(cell);
    });
    box.appendChild(row);

    const seenAs = groupsText(p);
    if (seenAs) {
      const line = el(doc, "div", "text-[10px] text-sky-300", seenAs);
      line.setAttribute("data-person-groups", String(p.individual_ids.length));
      box.appendChild(line);
    }
    const boxLine = boxesText(p, hasFrame);
    if (boxLine) {
      const line = el(doc, "div", "text-[10px] text-gray-400");
      const boxes = Array.isArray(p.boxes) ? p.boxes : [];
      if (boxes.length) {
        line.appendChild(el(doc, "span", "", boxes.length === 1 ? "Box " : "Boxes "));
        boxes.forEach((b, i) => {
          if (i) line.appendChild(el(doc, "span", "", ", "));
          const n = el(doc, "span", `font-bold ${BOX_TEXT[b.colour] || "text-blue-400"}`, String(b.n));
          n.setAttribute("data-person-box", String(b.n));
          line.appendChild(n);
        });
        line.appendChild(el(doc, "span", "", " on the big picture."));
      } else {
        line.textContent = boxLine;
      }
      line.setAttribute("data-person-boxes", boxes.map((b) => b.n).join(","));
      box.appendChild(line);
    }
    if (p.label && p.label_copied) {
      box.appendChild(el(doc, "div", "text-[10px] text-gray-500", "Name carried over from a same-person answer."));
    }
    const said = actionLine(p.story_action);
    if (said) {
      const line = el(doc, "div", "text-[10px] text-emerald-300", said);
      line.setAttribute("data-person-said", p.story_action.action);
      box.appendChild(line);
    }
    const why = whyLine(p.story_action);
    if (why) {
      const line = el(doc, "div", "text-[10px] text-gray-400 whitespace-pre-line", why);
      line.setAttribute("data-person-why", "1");
      box.appendChild(line);
    }

    const sending = ctx.flushing && ctx.flushing.story_id === story.story_id && ctx.flushing.person_key === p.individual_id
      ? ctx.flushing : null;
    if (sending) {
      const line = el(doc, "div", "rounded border border-gray-600 px-2 py-1 text-xs text-gray-300",
        `Saving: ${pendingText(p, sending.spec)} ...`);
      line.setAttribute("data-person-saving", sending.spec.name);
      box.appendChild(line);
    }
    const held = ctx.pending && ctx.pending.story_id === story.story_id && ctx.pending.person_key === p.individual_id
      ? ctx.pending : null;
    if (held) {
      // A name picked from the list: one tap, held until the next person, undoable until then.
      const line = el(doc, "div", "flex flex-wrap items-center gap-2 rounded border border-amber-500/60 bg-amber-900/20 px-2 py-1 text-xs text-amber-200");
      line.setAttribute("data-person-pending", held.spec.name);
      line.appendChild(el(doc, "span", "", `${pendingText(p, held.spec)}. Saved when you go to the next person.`));
      const undo = btnEl(doc, "Undo", "data-person-undo", "1", false);
      undo.addEventListener("click", () => ctx.undo(story));
      line.appendChild(undo);
      box.appendChild(line);
    }

    const whyInput = el(doc, "textarea", "w-full min-h-[2.5rem] rounded border border-gray-600 bg-gray-950 px-2 py-1 text-xs text-gray-100");
    whyInput.setAttribute("rows", "2");
    whyInput.setAttribute("maxlength", String(WHY_MAX_CHARS));
    whyInput.setAttribute("placeholder", "How could you tell? (optional) e.g. pushing a stroller, he has no young kids, not his build");
    whyInput.setAttribute("aria-label", `How you could tell, for person ${entry.letter}`);
    whyInput.setAttribute("data-person-why-input", "1");
    const send = (spec) => ctx.answer(story, p, Object.assign({ why: whyInput.value }, spec), box);

    const buttons = el(doc, "div", "flex flex-wrap items-center gap-2");
    if (p.label) {
      const yes = btnEl(doc, `Yes, that's ${p.label}`, "data-person-action", "yes", true);
      yes.addEventListener("click", () => send(answerSpec(p, "yes")));
      buttons.appendChild(yes);
    }
    const no = btnEl(doc, p.label ? "No, it's ..." : "It's ...", "data-person-action", "no", !p.label);
    buttons.appendChild(no);
    const cant = btnEl(doc, "Can't see them", "data-person-action", "cant_see", false);
    cant.setAttribute("title", "Too small, too dark or turned away. Orion won't ask about these pictures again.");
    cant.addEventListener("click", () => send(answerSpec(p, "cant_see")));
    buttons.appendChild(cant);
    box.appendChild(buttons);

    // "No, it's ...": the names Orion knows, someone not listed, I don't know who.
    const picker = el(doc, "div", "hidden space-y-2 rounded border border-gray-700 p-2");
    picker.setAttribute("data-person-picker", "1");
    const known = el(doc, "div", "flex flex-wrap gap-2");
    nameChoices(p, ctx.names).forEach((name) => {
      const b = btnEl(doc, name, "data-person-pick", name, false);
      b.addEventListener("click", () => {
        ctx.answer(story, p, Object.assign({ why: whyInput.value }, answerSpec(p, "pick", name)), box, true);
      });
      known.appendChild(b);
    });
    if (known.children.length) picker.appendChild(known);
    const other = el(doc, "div", "flex flex-wrap items-center gap-2");
    const input = el(doc, "input", "min-h-[2.25rem] w-full sm:w-40 rounded border border-gray-600 bg-gray-950 px-2 text-xs text-gray-100");
    input.setAttribute("type", "text");
    input.setAttribute("maxlength", "120");
    input.setAttribute("placeholder", "Someone not listed: type a name");
    input.setAttribute("aria-label", `Name for person ${entry.letter}`);
    input.setAttribute("data-person-name-input", "1");
    other.appendChild(input);
    const save = btnEl(doc, "Save name", "data-person-action", "typed", false);
    save.addEventListener("click", () => {
      const spec = answerSpec(p, "typed", input.value);
      if (!spec.name) { ctx.say("Type a name first."); return; }
      send(spec);
    });
    other.appendChild(save);
    picker.appendChild(other);
    const dunno = btnEl(doc, "I don't know who", "data-person-action", "dont_know", false);
    dunno.addEventListener("click", () => send(answerSpec(p, "dont_know")));
    picker.appendChild(dunno);
    const pickerId = `walkway-picker-${String(story.story_id).slice(0, 12)}-${entry.letter}`;
    picker.setAttribute("id", pickerId);
    no.setAttribute("aria-controls", pickerId);
    no.setAttribute("aria-expanded", "false");
    no.addEventListener("click", () => {
      const open = picker.classList.contains("hidden");
      picker.classList.toggle("hidden");
      no.setAttribute("aria-expanded", open ? "true" : "false");
    });
    box.appendChild(picker);
    box.appendChild(whyInput);
    if (seenAs) box.appendChild(renderNotSame(doc, story, p, ctx.onNotSame));
    return box;
  }

  // "These are N camera groups - not the same person?" (Orion's question 1): a wrong link
  // would otherwise hide inside one row. Opens one button per direct link.
  function renderNotSame(doc, story, p, onNotSame) {
    const wrap = el(doc, "div", "space-y-1");
    wrap.setAttribute("data-person-not-same", "1");
    const toggle = el(doc, "button", "text-[10px] text-gray-400 underline",
      `These are ${p.individual_ids.length} camera groups - not the same person?`);
    toggle.setAttribute("type", "button");
    toggle.setAttribute("data-person-not-same-toggle", "1");
    const list = el(doc, "div", "hidden flex flex-wrap gap-2");
    list.setAttribute("data-person-not-same-list", "1");
    const choices = notSameChoices(p);
    if (!choices.length) {
      list.appendChild(el(doc, "div", "text-[10px] text-gray-500",
        "They are joined through other camera groups, not directly. This card cannot split them."));
    }
    choices.forEach((c) => {
      const btn = el(doc, "button", "min-h-[2.25rem] rounded-lg border border-gray-600 bg-gray-900 px-3 py-1 text-xs text-gray-200 hover:bg-gray-700", c.label);
      btn.setAttribute("type", "button");
      btn.setAttribute("data-person-not-same-pair", `${c.a}|${c.b}`);
      btn.addEventListener("click", () => { if (typeof onNotSame === "function") onNotSame(story, p, c, wrap); });
      list.appendChild(btn);
    });
    toggle.addEventListener("click", () => list.classList.toggle("hidden"));
    wrap.appendChild(toggle);
    wrap.appendChild(list);
    return wrap;
  }

  function whyLine(a) {
    const why = a && typeof a.why === "string" ? a.why.trim() : "";
    return why ? `How you could tell: ${why}` : "";
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

  // ``ctx`` (the mounted card's state and actions): {answer, undo, go, say, onNotSame, cursor, pending, names}.
  function renderStory(doc, story, onVerdict, ctx) {
    const item = el(doc, "div", "rounded-lg border border-gray-700 bg-gray-800/60 p-3 space-y-2");
    item.setAttribute("data-story-id", story.story_id);
    item.__story = story;
    const shown = story.display_sentence || story.sentence;
    const sentence = el(doc, "div", "text-sm text-gray-100", shown);
    sentence.setAttribute("data-story-sentence", "1");
    item.appendChild(sentence);
    if (story.display_sentence && story.display_sentence !== story.sentence) {
      const was = el(doc, "div", "text-[10px] text-gray-500", `Orion first wrote: ${story.sentence}`);
      was.setAttribute("data-story-first-written", "1");
      item.appendChild(was);
    }
    if (story.reopened) {
      // Juniper's answer 5: a story she could only answer "not sure" on is asked again, once.
      const again = el(doc, "div", "text-[11px] text-amber-300",
        "Back for one more look: last time you could only say \"not sure\" here.");
      again.setAttribute("data-story-reopened", "1");
      item.appendChild(again);
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
    const entries = storyPeople(story);
    const owners = cropOwners(entries);
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
        `Picture from ${localTime(model.frame.t)}. Boxes: 1 = first (green), last = red. ` +
        "Each person below says which boxes are theirs."));
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
          const owner = c.crop_id ? owners[c.crop_id] : null;
          const label = el(doc, "span", "flex items-center gap-1 text-[10px] text-gray-500");
          if (owner) {
            label.appendChild(chipEl(doc, owner));
            cell.setAttribute("data-story-crop-person", owner.letter);
          }
          label.appendChild(el(doc, "span", "", shortTime(c.t)));
          cell.appendChild(label);
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

    const who = el(doc, "div", "space-y-1");
    who.setAttribute("data-story-people", String(entries.length));
    const askable = entries.filter((e) => isPictured(e.person));
    who.appendChild(el(doc, "div", "text-[11px] font-semibold text-gray-300",
      entries.length
        ? `Who is in this story? ${entries.length} ${entries.length === 1 ? "person" : "people"} by Orion's count, ` +
          "one at a time."
        : "No individual linked to these boxes yet."));
    if (entries.length && askable.length < entries.length) {
      const skipped = entries.length - askable.length;
      const line = el(doc, "div", "text-[10px] text-gray-500",
        `${skipped} without a picture of their own ${skipped === 1 ? "is" : "are"} not asked.`);
      line.setAttribute("data-story-not-asked", String(skipped));
      who.appendChild(line);
    }
    const c = ctx || {};
    if (typeof c.answer === "function" && askable.length) {
      const at = cursorIndex(askable, c.cursor && c.cursor.get(story.story_id));
      const progress = el(doc, "div", "text-[10px] text-gray-400", `Person ${at + 1} of ${askable.length}`);
      progress.setAttribute("data-story-progress", `${at + 1}/${askable.length}`);
      who.appendChild(progress);
      who.appendChild(renderAsk(doc, story, askable[at], !!model.frame, c));
      const nav = el(doc, "div", "flex flex-wrap items-center gap-2");
      if (at > 0) {
        const back = btnEl(doc, "Previous person", "data-story-nav", "back", false);
        back.addEventListener("click", () => c.go(story, askable[at - 1].person.individual_id));
        nav.appendChild(back);
      }
      if (at < askable.length - 1) {
        const next = btnEl(doc, `Next person (${at + 2} of ${askable.length})`, "data-story-nav", "next", true);
        next.addEventListener("click", () => c.go(story, askable[at + 1].person.individual_id));
        nav.appendChild(next);
      } else {
        const done = el(doc, "div", "text-[11px] text-gray-400", "That's everyone in this story.");
        done.setAttribute("data-story-nav-done", "1");
        nav.appendChild(done);
        const held = c.pending && c.pending.story_id === story.story_id;
        if (held) {
          // The last person has no "Next": "Done" sends the held pick.
          const save = btnEl(doc, "Done", "data-story-nav", "done", true);
          save.addEventListener("click", () => c.go(story, askable[at].person.individual_id));
          nav.appendChild(save);
        }
      }
      who.appendChild(nav);
    }
    item.appendChild(who);

    const parts = storyParts(story);
    if (parts && parts.length && ctx && typeof ctx.onPart === "function") {
      item.appendChild(renderParts(doc, story, parts, ctx));
    }
    const row = el(doc, "div", "flex flex-wrap gap-2");
    // "Can't tell" on every story: an unclear daylight picture must not be forced into "wrong".
    // With the parts recorded, it is the only whole-story button; without, the old four stay.
    const buttons = (parts && parts.length && ctx && typeof ctx.onPart === "function" ? [] : VERDICTS)
      .concat([CANT_TELL]);
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

  // "Right people? / direction? / place?" with Yes / No / Can't tell each.
  function renderParts(doc, story, parts, ctx) {
    const box = el(doc, "div", "space-y-1 rounded border border-gray-700 p-2");
    box.setAttribute("data-story-parts", parts.join(","));
    box.appendChild(el(doc, "div", "text-[11px] font-semibold text-gray-300", "Is the sentence right? One part at a time:"));
    const answers = (story.parts && story.parts.answers) || {};
    parts.forEach((part) => {
      const row = el(doc, "div", "flex flex-wrap items-center gap-2");
      row.setAttribute("data-story-part", part);
      row.appendChild(el(doc, "span", "min-w-[7.5rem] text-xs text-gray-200", PART_QUESTIONS[part]));
      const said = answers[part] || null;
      PART_ANSWERS.forEach((a) => {
        const btn = el(doc, "button", "", a.label);
        btn.setAttribute("type", "button");
        btn.setAttribute("data-part-verdict", a.value);
        setPressed(btn, !!said && said.verdict === a.value);
        btn.addEventListener("click", () => ctx.onPart(story, part, a.value, box));
        row.appendChild(btn);
      });
      if (said && said.source === "name_correction") {
        const note = el(doc, "span", "text-[10px] text-amber-300", "(from your name correction)");
        note.setAttribute("data-part-auto", "1");
        row.appendChild(note);
      }
      box.appendChild(row);
    });
    return box;
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

    // One-at-a-time state: which person each story shows, the one held answer, the known names.
    const cursor = new Map();          // story_id -> the person_key it shows
    let pending = null;                // a name picked from the list, held for Undo
    let flushing = null;               // that answer while it is being sent (Undo no longer applies)
    const names = [];
    const ctx = { cursor, names, answer, undo, go, say, onNotSame, onPart,
      get pending() { return pending; }, get flushing() { return flushing; } };

    function say(text) { if (status) status.textContent = text; }

    function rerender(storyId, fresh, focus) {
      const item = list.querySelector(`[data-story-id="${storyId}"]`);
      if (!item || !fresh) return;
      const next = renderStory(doc, fresh, onVerdict, ctx);
      item.replaceWith(next);
      if (focus) {
        const target = next.querySelector(focus);
        if (target && typeof target.focus === "function") target.focus();
      }
    }

    function storyOf(storyId) {
      const item = list.querySelector(`[data-story-id="${storyId}"]`);
      return item && item.__story ? item.__story : null;
    }

    // Send the held answer (a name picked from the list), if any. ``keepalive``: the page is closing.
    // Returns true when nothing is held any more. A failed send (not a 409: that reloads the card)
    // puts the answer back, so it is never lost and she can try again or Undo.
    async function flush(keepalive) {
      if (!pending) return true;
      const held = pending;
      pending = null;
      if (keepalive) {
        send(held.story, held.person, held.spec, null, true);
        return true;
      }
      flushing = held;
      rerender(held.story_id, storyOf(held.story_id) || held.story);
      let result = false;
      try {
        result = await send(held.story, held.person, held.spec, null, false);
      } finally {
        flushing = null;
      }
      if (result === false && !pending) {
        pending = held;
        rerender(held.story_id, storyOf(held.story_id) || held.story);
      }
      return result === true;
    }

    function undo(story) {
      if (!pending || pending.story_id !== story.story_id) {
        say("Already saving that answer; to change it, answer this person again.");
        return;
      }
      pending = null;
      say("Undone: nothing was saved.");
      rerender(story.story_id, storyOf(story.story_id) || story, "[data-person-question]");
    }

    async function go(story, personKey) {
      if (!(await flush(false))) return;
      cursor.set(story.story_id, personKey);
      rerender(story.story_id, storyOf(story.story_id) || story, "[data-person-question]");
    }

    async function answer(story, person, spec, box, hold) {
      const samePerson = pending && pending.story_id === story.story_id && pending.person_key === person.individual_id;
      if (samePerson) {
        // A new answer about the same person replaces the held pick: never two contradicting rows.
        pending = null;
      } else if (pending && !(await flush(false))) {
        return;
      }
      if (hold) {
        // One tap, no confirm (Juniper's answer 1): held as "Changed to X - Undo" until the next person.
        pending = { story, person, spec, story_id: story.story_id, person_key: person.individual_id };
        cursor.set(story.story_id, person.individual_id);
        say(`${pendingText(person, spec)}. Saved when you go to the next person; Undo until then.`);
        rerender(story.story_id, storyOf(story.story_id) || story, "[data-person-undo]");
        return;
      }
      cursor.set(story.story_id, person.individual_id);
      await send(story, person, spec, box, false);
    }

    async function send(story, person, spec, box, keepalive) {
      const buttons = box ? box.querySelectorAll("[data-person-action], [data-person-pick]") : [];
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetchFn(
          `/api/vision/motion-stories/${encodeURIComponent(story.story_id)}/people/${encodeURIComponent(person.individual_id)}`, {
            method: "POST", headers: { "Content-Type": "application/json" }, keepalive: !!keepalive,
            // A yes carries the name the card showed: the server refuses it if the name changed since.
            // A linked person (`grouped`) is answered for each of its camera groups, on the pictures shown.
            body: JSON.stringify({ action: spec.action, name: spec.name || "", why: String(spec.why || "").trim(),
              group: !!person.grouped, card: CARD, dont_know: !!spec.dontKnow,
              crop_ids: person.grouped ? personPictures(person).map((c) => c.crop_id).filter(Boolean) : undefined }),
          });
        if (keepalive) return true;
        if (res.status === 409) {
          // The name or the pictures changed since the card loaded (renamed, or linked to another
          // camera group): show the new card, ask again.
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (_) { /* not JSON */ }
          await load(false, true);
          say(detail === "pictures_changed"
            ? "Orion's pictures of this person changed since this card loaded. Showing the new ones; nothing saved."
            : "That name changed since this card loaded. Showing the new one; nothing saved.");
          return "conflict";
        }
        if (!res.ok) {
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (_) { /* not JSON */ }
          throw new Error(detail ? `${res.status} ${detail}` : `HTTP ${res.status}`);
        }
        const body = await res.json();
        if (body.tally) showTally(body.tally);
        // The server's fresh story; else the one shown (so a replaced held pick leaves the card).
        rerender(story.story_id, body.story || storyOf(story.story_id) || story);
        let text = spec.action === "name" ? `Saved: ${spec.name}. Orion applies it like any "who is this?" answer.`
          : spec.action === "confirm" ? `Saved: yes, that's ${person.label || "them"}.`
          : spec.action === "someone_else"
            ? `Saved: not ${person.label || "them"}${spec.name ? `, it's ${spec.name}` : ""}. Orion will split those pictures off.`
          : spec.dontKnow ? "Saved: you don't know who this is."
          : "Saved: can't see them. Orion won't ask about these pictures again.";
        if (body.skipped_all) text = "Already recorded: you said the same before. Nothing new saved.";
        if (body.wrong_name_recorded) text += " Recorded as a wrong name for this story.";
        say(text);
        return true;
      } catch (err) {
        say(`Could not save that (${err.message}). Try again.`);
        buttons.forEach((b) => { b.disabled = false; });
        return false;
      }
    }

    async function onPart(story, part, verdict, box) {
      const buttons = box.querySelectorAll("[data-part-verdict]");
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetchFn(`/api/vision/motion-stories/${encodeURIComponent(story.story_id)}/parts`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ part, verdict }),
        });
        if (!res.ok) {
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (_) { /* not JSON */ }
          throw new Error(detail ? `${res.status} ${detail}` : `HTTP ${res.status}`);
        }
        const body = await res.json();
        if (body.tally) showTally(body.tally);
        rerender(story.story_id, body.story || storyOf(story.story_id) || story);
        const left = (storyParts(body.story || story) || []).filter((p) => {
          const a = body.story && body.story.parts && body.story.parts.answers ? body.story.parts.answers[p] : null;
          return !a;
        }).length;
        say(body.story_verdict
          ? `Saved. Every part answered: this story counts as ${VERDICT_WORDS[body.story_verdict] || body.story_verdict}.`
          : `Saved. ${left} more ${left === 1 ? "part" : "parts"} to answer before this story counts.`);
      } catch (err) {
        say(`Could not save that (${err.message}). Try again.`);
        buttons.forEach((b) => { b.disabled = false; });
      }
    }

    async function onNotSame(story, person, choice, wrap) {
      const buttons = wrap.querySelectorAll("[data-person-not-same-pair]");
      buttons.forEach((b) => { b.disabled = true; });
      try {
        const res = await fetchFn(
          `/api/vision/motion-stories/${encodeURIComponent(story.story_id)}/persons/${encodeURIComponent(person.person_key)}/not-same`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ a: choice.a, b: choice.b }),
          });
        if (!res.ok) {
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (_) { /* not JSON */ }
          if (detail === "still_linked") {
            if (status) status.textContent = "Not saved: those two are also joined through other camera groups, so this answer would not split them.";
            buttons.forEach((b) => { b.disabled = false; });
            return;
          }
          throw new Error(detail ? `${res.status} ${detail}` : `HTTP ${res.status}`);
        }
        const body = await res.json();
        const item = wrap.closest("[data-story-id]");
        if (body.story && item) item.replaceWith(renderStory(doc, body.story, onVerdict, ctx));
        if (status) status.textContent = "Saved: not the same person. Orion unlinks them on its next pass; they then show as two people.";
      } catch (err) {
        if (status) status.textContent = `Could not save that (${err.message}). Try again.`;
        buttons.forEach((b) => { b.disabled = false; });
      }
    }

    async function loadNames() {
      try {
        const res = await fetchFn(`/api/vision/known-names?limit=${NAMES_LIMIT}`);
        if (!res.ok) return;
        const body = await res.json();
        names.splice(0, names.length, ...(Array.isArray(body.names) ? body.names : []));
      } catch (_) { /* the list is optional: typing a name still works */ }
    }

    async function load(fromTimer, skipFlush) {
      // The 60 s refresh never runs over an answer held for Undo; anything else sends it first.
      if (fromTimer === true && (pending || flushing)) return;
      if (!skipFlush && pending && !(await flush(false))) return;
      if (!fromTimer) await loadNames();
      const unchecked = uncheckedBox && uncheckedBox.checked ? "true" : "false";
      const checkable = !checkableBox || checkableBox.checked ? "true" : "false";
      try {
        const res = await fetchFn(
          `/api/vision/motion-stories?limit=20&unchecked_only=${unchecked}&checkable_only=${checkable}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        showTally(body.tally);
        // The 60 s refresh never wipes a name Juniper is typing, nor an answer held for Undo
        // (a filter change she makes does reload, and sends the held answer first).
        const typing = fromTimer === true && Array.from(
          list.querySelectorAll("[data-person-name-input], [data-person-why-input]"))
          .some((i) => i.value.trim() || doc.activeElement === i);
        if (typing) return;
        list.textContent = "";
        const stories = body.stories || [];
        stories.forEach((s) => list.appendChild(renderStory(doc, s, onVerdict, ctx)));
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

    // Leaving the page sends the held answer (a keepalive request survives the unload).
    if (typeof global.addEventListener === "function") {
      global.addEventListener("pagehide", () => { flush(true); });
      // Mobile browsers may never fire pagehide when she switches apps: hidden sends it too.
      global.addEventListener("pageshow", (ev) => { if (ev && ev.persisted) load(false); });
    }
    if (typeof doc.addEventListener === "function") {
      doc.addEventListener("visibilitychange", () => { if (doc.visibilityState === "hidden") flush(true); });
    }
    if (select) select.addEventListener("change", sync);
    if (uncheckedBox) uncheckedBox.addEventListener("change", () => load(false));
    if (checkableBox) checkableBox.addEventListener("change", () => load(false));
    sync();
    return { load, sync, flush };
  }

  const api = { VIEW_VALUE, VERDICTS, CANT_TELL, tallyLine, gateLine, pickThumbs, stripModel, hasPicture,
                personTitle, actionLine, whyLine, handoffText, renderStory, mount,
                storyPeople, personPictures, isPictured, cropOwners, boxesText, groupsText, notSameChoices,
                questionText, nameChoices, answerSpec, pendingText, cursorIndex, sourceText, storyParts };

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

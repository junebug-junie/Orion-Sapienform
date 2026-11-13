(function (global) {
  // "Orion is asking" card in the Vision panel (walkway camera idea 3,
  // docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md).
  // Lists open questions from GET /api/asks and posts Juniper's answer or
  // dismissal to /api/asks/{id}/answer | /dismiss (scripts/ask_routes.py).
  //
  // Polls instead of listening on a socket: the Hub has no push channel for
  // panels, and the orion_ask table is the only truth either way.
  //
  // Everything is rendered with textContent / DOM attributes, never
  // innerHTML: question text and image refs come from the database.
  //
  // Memory confirmation cards (2026-10-06, memory-episode spec sections 3/5):
  // a card whose source_kind is in RESOLVABLE_KINDS gets Confirm / Revise /
  // Reject instead of Answer / Dismiss, and posts to /api/asks/{id}/resolve.
  // Revise opens a box prefilled with the memory's current wording; it cannot
  // be sent empty. The panel is mounted at the top of the Hub home.
  //
  // Visual confirmation cards (2026-10-15): every picture in image_refs is
  // shown side by side (older rows have only image_ref). A "same person?"
  // card (source_kind vision_same_individual) shows both faces and answers
  // with Same person (yes) / Different people (no) / Not sure (dismiss).
  //
  // Names on visual cards (2026-10-15, Juniper): a "same person?" card is a
  // choice plus one Send button. "Same person" opens one optional name box for
  // both; "Different people" opens one optional name box under each face; the
  // names ride along as {names: {<individual_id>: name}}. A "who is this?" card
  // with several pictures asks, under every picture after the first, whether
  // it is the same person (the default), someone else (optionally named) or
  // not sure; that rides along as {pictures: [...]}. The main box still names
  // the first picture, which is the one the question is about.
  //
  // "I don't know" on "who is this?" cards (2026-10-27, Juniper): typed
  // non-answers ("no", "unsure") were being saved as names. The card now has an
  // explicit "I don't know" button. It is a dismissal that says why
  // (POST /dismiss {reason: "dont_know"}): no name is stored, and Orion may ask
  // again after the cooldown. Plain Dismiss still means "never ask again".
  // There is no word list on typed answers: the button is the fix.
  const RESOLVABLE_KINDS = ["memory_confirmation", "open_question"];

  function isResolvable(ask) {
    return !!ask && RESOLVABLE_KINDS.indexOf(String(ask.source_kind || "")) >= 0;
  }

  const POLL_MS = 60000;

  // "thumb:<sha256>" is orion-vision-host's crop thumbnail (one embedded box,
  // never the patio). The Hub serves it by hash from a read-only mount.
  const THUMB_REF = /^thumb:([0-9a-f]{64})$/;

  function askImageView(imageRef) {
    const ref = typeof imageRef === "string" ? imageRef.trim() : "";
    if (!ref) return { kind: "none", ref: "" };
    const thumb = THUMB_REF.exec(ref);
    if (thumb) return { kind: "img", ref: "/api/vision/crop-thumbs/" + thumb[1] };
    // Otherwise only a real web URL can be shown as a picture. Anything else
    // (a path on the vision host's disk, an embedding ref) is not something
    // this Hub can serve, so it is shown as text rather than a broken image.
    if (/^https?:\/\//i.test(ref)) return { kind: "img", ref: ref };
    return { kind: "text", ref: ref };
  }

  const SAME_PERSON_KIND = "vision_same_individual";
  // "Is this <name>?" (orion-sql-writer person learning, 2026-10-20): yes / no,
  // "Can't tell" is a dismissal. source_ref is "<individual_id>|<name>".
  const PERSON_CONFIRM_KIND = "vision_person_confirm";
  const WHO_IS_THIS_KIND = "vision_individual";
  // More than this many pictures would not fit a phone-width card.
  const MAX_IMAGES = 4;

  function isSamePerson(ask) {
    return !!ask && String(ask.source_kind || "") === SAME_PERSON_KIND;
  }

  function isPersonConfirm(ask) {
    return !!ask && String(ask.source_kind || "") === PERSON_CONFIRM_KIND;
  }

  // The name an "Is this <name>?" card asks about ("" when the ref has none).
  function personConfirmName(ask) {
    const ref = String((ask && ask.source_ref) || "");
    const i = ref.indexOf("|");
    return i > 0 ? ref.slice(i + 1).trim() : "";
  }

  // What an "Is this <name>?" card sends: yes / no, or a dismissal for "Can't tell".
  function personConfirmSubmission(choice) {
    if (choice === "yes" || choice === "no") return { action: "answer", answer: choice, extra: null };
    if (choice === "unsure") return { action: "dismiss", answer: "", extra: null };
    return null;
  }

  // What an "Is this <name>?" card with several pictures sends (2026-10-23):
  // choices[i] is "same" (it is <name>), "different" (it is not) or "unsure".
  // At least one <name> -> yes; none, but at least one "not" -> no; all "not
  // sure" -> null (nothing to send). Every picture rides along, so a card holding two
  // people can say which picture is <name>.
  function personConfirmPicturesSubmission(views, choices) {
    if (!views || views.length < 2) return null;
    const pictures = views.map(function (v, i) {
      return { image_ref: v.raw, verdict: choices[i] || "unsure" };
    });
    const verdicts = pictures.map(function (p) { return p.verdict; });
    if (verdicts.indexOf("same") >= 0) return { action: "answer", answer: "yes", extra: { pictures: pictures } };
    if (verdicts.indexOf("different") >= 0) return { action: "answer", answer: "no", extra: { pictures: pictures } };
    // Nothing chosen: refused, never a silent dismissal ("Can't tell" is its own button).
    return null;
  }

  // Every picture an ask shows, in order, no repeats: image_refs, else the
  // single image_ref of an older row.
  function askImageViews(ask) {
    const a = ask || {};
    const refs = Array.isArray(a.image_refs) && a.image_refs.length ? a.image_refs : [a.image_ref];
    const out = [];
    const seen = {};
    refs.forEach(function (r) {
      const view = askImageView(r);
      if (view.kind === "none" || seen[view.kind + ":" + view.ref]) return;
      seen[view.kind + ":" + view.ref] = true;
      // raw: the ref exactly as stored, which is what an answer names a picture by.
      out.push({ kind: view.kind, ref: view.ref, raw: String(r).trim() });
    });
    return out.slice(0, MAX_IMAGES);
  }

  function imageAlt(index, total, samePerson) {
    if (samePerson && total === 2) return index === 0 ? "Left face Orion is asking about" : "Right face Orion is asking about";
    if (total <= 1) return "What Orion is asking about";
    return "Picture " + (index + 1) + " of " + total + " of what Orion is asking about";
  }

  // The two individual ids of a "same person?" card, left then right
  // (source_ref is "<idA>|<idB>", the same order as image_refs).
  function pairIds(ask) {
    const parts = String((ask && ask.source_ref) || "").split("|");
    return parts.length === 2 && parts[0] && parts[1] ? parts : [];
  }

  function askCardViewModel(ask) {
    const a = ask || {};
    const images = askImageViews(a);
    return {
      pairIds: isSamePerson(a) ? pairIds(a) : [],
      whoIsThis: String(a.source_kind || "") === WHO_IS_THIS_KIND,
      askId: String(a.ask_id || ""),
      question: String(a.question || ""),
      image: images[0] || { kind: "none", ref: "" },
      images: images,
      samePerson: isSamePerson(a),
      personConfirm: isPersonConfirm(a),
      personName: isPersonConfirm(a) ? personConfirmName(a) : "",
      askedAt: a.created_at ? String(a.created_at) : "",
      about: a.source_kind ? String(a.source_kind) + ": " + String(a.source_ref || "") : "",
      resolvable: isResolvable(a),
      statement: a.memory_statement ? String(a.memory_statement) : "",
    };
  }

  function statusLine(asks) {
    const n = Array.isArray(asks) ? asks.length : 0;
    if (n === 0) return "Orion has no open questions for you.";
    if (n === 1) return "Orion has 1 question for you.";
    return "Orion has " + n + " questions for you.";
  }

  // Mirrors orion.memory.episode.confirmation.MIN_REVISION_WORDS / revision_problem: structural
  // checks only. A note that only says "no" belongs on Reject, which the hint says.
  const MIN_REVISION_WORDS = 6;

  function revisionProblem(note, current) {
    const words = String(note || "").split(/\s+/).filter(Boolean);
    if (words.length === 0) return "revised_needs_note";
    if (words.length < MIN_REVISION_WORDS) return "revised_too_short";
    const norm = function (t) { return String(t || "").split(/\s+/).filter(Boolean).join(" ").toLowerCase(); };
    if (current && norm(note) === norm(current)) return "revised_unchanged";
    return null;
  }

  function errorLine(status, detail) {
    if (detail === "revised_needs_note") return "Write how I should remember it first.";
    if (detail === "revised_too_short") return "Write it as a full sentence (at least " + MIN_REVISION_WORDS + " words). To drop it, press Reject.";
    if (detail === "revised_unchanged") return "That is the same as what I have. Change it, or press Confirm.";
    if (detail === "answer_must_be_yes_or_no") return "Press Same person, Different people, or Not sure.";
    if (detail === "same_person_takes_one_name") return "Same person takes one name.";
    if (detail === "name_too_long") return "That name is too long (120 characters at most).";
    if (detail === "dont_know_only_on_who_is_this_cards") return "I don't know is only for \"who is this?\" questions.";
    if (detail === "pick_an_answer") return "Pick Same person, Different people, or Not sure first.";
    if (status === 409) return "Someone already answered this one, or it expired.";
    if (status === 404) return "That question no longer exists.";
    if (status === 503) return "Can't reach the question store right now.";
    const text = typeof detail === "string" ? detail : "";
    return "Something went wrong" + (text ? ": " + text : ".");
  }

  function el(doc, tag, cls, text) {
    const node = doc.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderResolvable(doc, vm, onAction) {
    const card = el(doc, "div", "rounded-lg border border-violet-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.setAttribute("data-ask-kind", "resolvable");
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));
    const row = el(doc, "div", "flex items-center gap-2");
    function btn(label, action, cls) {
      const b = el(doc, "button", "text-xs rounded px-3 py-1 " + cls, label);
      b.setAttribute("type", "button");
      b.setAttribute("data-ask-action", action);
      return b;
    }
    const confirmBtn = btn("Confirm", "confirm", "bg-emerald-700 hover:bg-emerald-600 text-white");
    const reviseBtn = btn("Revise", "revise", "bg-sky-700 hover:bg-sky-600 text-white");
    const rejectBtn = btn("Reject", "reject", "bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700");
    const revise = el(doc, "div", "space-y-1 hidden");
    revise.setAttribute("data-ask-revise", vm.askId);
    // The hidden attribute as well as the class: hidden must not depend on the stylesheet.
    revise.hidden = true;
    revise.appendChild(el(doc, "div", "text-[11px] text-gray-400",
      "How should I remember it? Rewrite it in full. If it should not be kept at all, press Reject instead."));
    const box = el(doc, "textarea", "w-full bg-gray-900 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700");
    box.setAttribute("maxlength", "500");
    box.setAttribute("rows", "3");
    box.setAttribute("data-ask-input", vm.askId);
    const saveBtn = btn("Save revision", "save-revision", "bg-sky-700 hover:bg-sky-600 text-white");
    revise.appendChild(box);
    revise.appendChild(saveBtn);
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    const buttons = [confirmBtn, reviseBtn, rejectBtn, saveBtn];
    function run(resolution, text) {
      buttons.forEach(function (b) { b.disabled = true; });
      return Promise.resolve(onAction(vm.askId, "resolve", text, note, resolution)).finally(function () {
        buttons.forEach(function (b) { b.disabled = false; });
      });
    }
    confirmBtn.addEventListener("click", function () { return run("confirmed", ""); });
    rejectBtn.addEventListener("click", function () { return run("rejected", ""); });
    reviseBtn.addEventListener("click", function () {
      revise.classList.remove("hidden");
      revise.hidden = false;
      if (!String(box.value || "").trim()) box.value = vm.statement;
      if (box.focus) box.focus();
    });
    saveBtn.addEventListener("click", function () {
      const problem = revisionProblem(box.value, vm.statement);
      if (problem) {
        note.textContent = errorLine(422, problem);
        note.classList.remove("hidden");
        return;
      }
      return run("revised", box.value);
    });
    row.appendChild(confirmBtn);
    row.appendChild(reviseBtn);
    row.appendChild(rejectBtn);
    card.appendChild(row);
    card.appendChild(revise);
    card.appendChild(note);
    return card;
  }

  // Layout-critical styles are set inline as well as by class: side by side
  // must not depend on the Tailwind CDN having loaded.
  function setStyle(node, styles) {
    if (!node.style) return;
    Object.keys(styles).forEach(function (k) { node.style[k] = styles[k]; });
  }

  // The pictures, side by side. Each one shrinks to share the card width, so
  // two faces fit a phone screen; a single picture keeps its old size.
  // below(index, view): optional controls shown under that picture.
  function renderImages(doc, vm, below) {
    const row = el(doc, "div", "flex gap-2 items-start");
    setStyle(row, { display: "flex", gap: "0.5rem", alignItems: "flex-start" });
    row.setAttribute("data-ask-images", String(vm.images.length));
    const total = vm.images.length;
    vm.images.forEach(function (view, i) {
      if (view.kind === "img") {
        const fig = el(doc, "figure", total > 1 ? "flex-1 min-w-0 m-0" : "m-0");
        const img = el(doc, "img", (total > 1 ? "w-full " : "") + "max-h-48 object-contain rounded border border-gray-700");
        setStyle(fig, total > 1 ? { flex: "1 1 0", minWidth: "0", margin: "0" } : { margin: "0" });
        setStyle(img, total > 1 ? { width: "100%", maxHeight: "12rem", objectFit: "contain" }
          : { maxWidth: "100%", maxHeight: "12rem" });
        img.setAttribute("src", view.ref);
        img.setAttribute("alt", imageAlt(i, total, vm.samePerson));
        img.setAttribute("loading", "lazy");
        const gone = el(doc, "div", "text-[11px] text-gray-500 hidden", "Picture no longer available.");
        gone.hidden = true;
        // A pruned or missing thumbnail must not leave a broken-image icon;
        // say so instead, so a "same person?" card never looks like one face.
        img.addEventListener("error", function () {
          img.hidden = true;
          gone.hidden = false;
          if (gone.classList) gone.classList.remove("hidden");
        });
        fig.appendChild(img);
        fig.appendChild(gone);
        const extra = below ? below(i, view) : null;
        if (extra) fig.appendChild(extra);
        row.appendChild(fig);
      } else {
        const box = el(doc, "div", "text-[11px] text-gray-500 break-all", "Picture: " + view.ref);
        const extra = below ? below(i, view) : null;
        if (extra) {
          const wrap = el(doc, "div", total > 1 ? "flex-1 min-w-0" : "");
          if (total > 1) setStyle(wrap, { flex: "1 1 0", minWidth: "0" });
          wrap.appendChild(box);
          wrap.appendChild(extra);
          row.appendChild(wrap);
        } else {
          row.appendChild(box);
        }
      }
    });
    return row;
  }

  let uid = 0;
  function nextId(prefix) {
    uid += 1;
    return prefix + "-" + uid;
  }

  // A labelled text box. The label is a real <label for>, so a screen reader
  // reads it and a tap on it focuses the box.
  function nameField(doc, askId, labelText, dataName) {
    const wrap = el(doc, "div", "space-y-1");
    const id = nextId("ask-name");
    const label = el(doc, "label", "block text-[11px] text-gray-400", labelText);
    label.setAttribute("for", id);
    const input = el(doc, "input", "w-full bg-gray-900 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700");
    setStyle(input, { width: "100%", boxSizing: "border-box", minWidth: "0" });
    input.setAttribute("id", id);
    input.setAttribute("type", "text");
    input.setAttribute("maxlength", "120");
    input.setAttribute("autocomplete", "off");
    input.setAttribute("data-ask-input", askId);
    if (dataName) input.setAttribute("data-ask-name", dataName);
    wrap.appendChild(label);
    wrap.appendChild(input);
    return { wrap: wrap, input: input };
  }

  function showNode(node, on) {
    node.hidden = !on;
    if (node.classList) {
      if (on) node.classList.remove("hidden");
      else node.classList.add("hidden");
    }
  }

  // What a "same person?" card sends for a choice and the typed names.
  // Not sure is a dismissal (no names); empty names are left out.
  function samePersonSubmission(choice, ids, sameName, leftName, rightName) {
    const clean = function (t) { return String(t || "").split(/\s+/).filter(Boolean).join(" "); };
    if (choice === "unsure") return { action: "dismiss", answer: "", extra: null };
    if (choice !== "same" && choice !== "different") return null;
    const names = {};
    if (ids.length === 2) {
      if (choice === "same" && clean(sameName)) {
        names[ids[0]] = clean(sameName);
        names[ids[1]] = clean(sameName);
      }
      if (choice === "different") {
        if (clean(leftName)) names[ids[0]] = clean(leftName);
        if (clean(rightName)) names[ids[1]] = clean(rightName);
      }
    }
    return {
      action: "answer",
      answer: choice === "same" ? "yes" : "no",
      extra: Object.keys(names).length ? { names: names } : null,
    };
  }

  function renderSamePerson(doc, vm, onAction) {
    const card = el(doc, "div", "rounded-lg border border-sky-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.setAttribute("data-ask-kind", "same-person");
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    const canName = vm.pairIds.length === 2;
    // One optional name box under each face, shown for "Different people".
    const faceFields = [];
    const faceWraps = [];
    if (vm.images.length) {
      card.appendChild(renderImages(doc, vm, function (i) {
        if (!canName || i > 1) return null;
        const f = nameField(doc, vm.askId, (i === 0 ? "Left" : "Right") + " face: who is it? (optional)",
          i === 0 ? "left" : "right");
        faceFields[i] = f.input;
        faceWraps[i] = f.wrap;
        showNode(f.wrap, false);
        return f.wrap;
      }));
    }
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));

    // The choice: three radio buttons in one group, then one Send.
    const group = el(doc, "fieldset", "flex flex-wrap items-center gap-3 border-0 p-0 m-0");
    setStyle(group, { border: "0", padding: "0", margin: "0" });
    group.setAttribute("data-ask-choices", vm.askId);
    const legend = el(doc, "legend", "sr-only", "Are these the same person?");
    // Visually hidden without the stylesheet too; still read by a screen reader.
    setStyle(legend, { position: "absolute", width: "1px", height: "1px", overflow: "hidden",
      clip: "rect(0 0 0 0)", whiteSpace: "nowrap" });
    group.appendChild(legend);
    const groupName = nextId("ask-choice");
    const radios = {};
    [["same", "Same person"], ["different", "Different people"], ["unsure", "Not sure"]].forEach(function (c) {
      const label = el(doc, "label", "inline-flex items-center gap-1 text-xs text-gray-200");
      const radio = el(doc, "input", "");
      radio.setAttribute("type", "radio");
      radio.setAttribute("name", groupName);
      radio.setAttribute("value", c[0]);
      radio.setAttribute("data-ask-choice", c[0]);
      radio.addEventListener("change", function () { choose(c[0]); });
      radios[c[0]] = radio;
      label.appendChild(radio);
      label.appendChild(el(doc, "span", "", c[1]));
      group.appendChild(label);
    });
    card.appendChild(group);

    // One name box for both, shown for "Same person".
    const same = canName ? nameField(doc, vm.askId, "Who is this? (optional)", "same") : null;
    if (same) {
      showNode(same.wrap, false);
      card.appendChild(same.wrap);
    }
    let choice = "";
    function choose(c) {
      choice = c;
      Object.keys(radios).forEach(function (k) { radios[k].checked = k === c; });
      if (same) showNode(same.wrap, c === "same");
      faceWraps.forEach(function (w) { if (w) showNode(w, c === "different"); });
    }

    const sendBtn = el(doc, "button", "text-xs rounded px-3 py-1 bg-sky-700 hover:bg-sky-600 text-white", "Send");
    sendBtn.setAttribute("type", "button");
    sendBtn.setAttribute("data-ask-action", "send");
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    sendBtn.addEventListener("click", function () {
      const sub = samePersonSubmission(choice, vm.pairIds, same ? same.input.value : "",
        faceFields[0] ? faceFields[0].value : "", faceFields[1] ? faceFields[1].value : "");
      if (!sub) {
        note.textContent = errorLine(422, "pick_an_answer");
        note.classList.remove("hidden");
        return;
      }
      sendBtn.disabled = true;
      return Promise.resolve(onAction(vm.askId, sub.action, sub.answer, note, undefined, sub.extra)).finally(function () {
        sendBtn.disabled = false;
      });
    });
    card.appendChild(sendBtn);
    card.appendChild(note);
    return card;
  }

  function renderPersonConfirm(doc, vm, onAction) {
    if (vm.images.length > 1) return renderPersonConfirmByPicture(doc, vm, onAction);
    const card = el(doc, "div", "rounded-lg border border-sky-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.setAttribute("data-ask-kind", "person-confirm");
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    if (vm.images.length) card.appendChild(renderImages(doc, vm, null));
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));

    const group = el(doc, "fieldset", "flex flex-wrap items-center gap-3 border-0 p-0 m-0");
    setStyle(group, { border: "0", padding: "0", margin: "0" });
    group.setAttribute("data-ask-choices", vm.askId);
    const legend = el(doc, "legend", "sr-only", vm.personName ? "Is this " + vm.personName + "?" : "Is this them?");
    setStyle(legend, { position: "absolute", width: "1px", height: "1px", overflow: "hidden",
      clip: "rect(0 0 0 0)", whiteSpace: "nowrap" });
    group.appendChild(legend);
    const groupName = nextId("ask-choice");
    let choice = "";
    const yesText = vm.personName ? "Yes, it is " + vm.personName : "Yes";
    [["yes", yesText], ["no", "No"], ["unsure", "Can't tell"]].forEach(function (c) {
      const label = el(doc, "label", "inline-flex items-center gap-1 text-xs text-gray-200");
      const radio = el(doc, "input", "");
      radio.setAttribute("type", "radio");
      radio.setAttribute("name", groupName);
      radio.setAttribute("value", c[0]);
      radio.setAttribute("data-ask-choice", c[0]);
      radio.addEventListener("change", function () { choice = c[0]; });
      label.appendChild(radio);
      label.appendChild(el(doc, "span", "", c[1]));
      group.appendChild(label);
    });
    card.appendChild(group);

    const sendBtn = el(doc, "button", "text-xs rounded px-3 py-1 bg-sky-700 hover:bg-sky-600 text-white", "Send");
    sendBtn.setAttribute("type", "button");
    sendBtn.setAttribute("data-ask-action", "send");
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    sendBtn.addEventListener("click", function () {
      const sub = personConfirmSubmission(choice);
      if (!sub) {
        note.textContent = errorLine(422, "pick_an_answer");
        note.classList.remove("hidden");
        return;
      }
      sendBtn.disabled = true;
      return Promise.resolve(onAction(vm.askId, sub.action, sub.answer, note, undefined, sub.extra)).finally(function () {
        sendBtn.disabled = false;
      });
    });
    card.appendChild(sendBtn);
    card.appendChild(note);
    return card;
  }

  // "Is this <name>?" with several pictures: one choice under each picture,
  // because the pictures can be of different people (2026-10-23, Juniper: "one
  // of the three is Emerald but I can't specify which").
  function renderPersonConfirmByPicture(doc, vm, onAction) {
    const name = vm.personName || "them";
    const card = el(doc, "div", "rounded-lg border border-sky-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.setAttribute("data-ask-kind", "person-confirm");
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    card.appendChild(el(doc, "div", "text-[11px] text-gray-400",
      "These pictures may not all be the same person. Say for each one."));
    const choices = {};
    card.appendChild(renderImages(doc, vm, function (i) {
      const wrap = el(doc, "div", "space-y-1");
      const id = nextId("ask-pic");
      const label = el(doc, "label", "block text-[11px] text-gray-400", "Picture " + (i + 1) + ":");
      label.setAttribute("for", id);
      const select = el(doc, "select", "w-full bg-gray-900 text-gray-100 text-xs rounded px-1 py-1 border border-gray-700");
      setStyle(select, { width: "100%", minWidth: "0", boxSizing: "border-box" });
      select.setAttribute("id", id);
      select.setAttribute("data-ask-picture", String(i));
      [["unsure", "Not sure"], ["same", name], ["different", "Not " + name]].forEach(function (o) {
        const opt = el(doc, "option", "", o[1]);
        opt.setAttribute("value", o[0]);
        select.appendChild(opt);
      });
      select.value = "unsure";
      choices[i] = "unsure";
      select.addEventListener("change", function () { choices[i] = String(select.value || "unsure"); });
      wrap.appendChild(label);
      wrap.appendChild(select);
      return wrap;
    }));
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));
    const sendBtn = el(doc, "button", "text-xs rounded px-3 py-1 bg-sky-700 hover:bg-sky-600 text-white", "Send");
    sendBtn.setAttribute("type", "button");
    sendBtn.setAttribute("data-ask-action", "send");
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    const cantTellBtn = el(doc, "button",
      "text-xs rounded px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700", "Can't tell");
    cantTellBtn.setAttribute("type", "button");
    cantTellBtn.setAttribute("data-ask-action", "dismiss");
    const buttons = [sendBtn, cantTellBtn];
    function run(sub) {
      buttons.forEach(function (b) { b.disabled = true; });
      return Promise.resolve(onAction(vm.askId, sub.action, sub.answer, note, undefined, sub.extra)).finally(function () {
        buttons.forEach(function (b) { b.disabled = false; });
      });
    }
    sendBtn.addEventListener("click", function () {
      const sub = personConfirmPicturesSubmission(vm.images, choices);
      if (!sub) {
        note.textContent = "Choose " + name + " or Not " + name + " for at least one picture, or press Can't tell.";
        note.classList.remove("hidden");
        return;
      }
      return run(sub);
    });
    cantTellBtn.addEventListener("click", function () { return run(personConfirmSubmission("unsure")); });
    const row = el(doc, "div", "flex items-center gap-2");
    row.appendChild(sendBtn);
    row.appendChild(cantTellBtn);
    card.appendChild(row);
    card.appendChild(note);
    return card;
  }

  // What a multi-picture "who is this?" card sends about pictures 2..n.
  // choices[i] is "same" | "different" | "unsure" for picture i (i >= 1).
  function pictureSubmission(views, choices, names) {
    if (!views || views.length < 2) return null;
    const out = [];
    for (let i = 1; i < views.length; i++) {
      const verdict = choices[i] || "same";
      const entry = { image_ref: views[i].raw, verdict: verdict };
      const name = String(names[i] || "").split(/\s+/).filter(Boolean).join(" ");
      if (verdict === "different" && name) entry.name = name;
      out.push(entry);
    }
    return out;
  }

  function renderAsk(doc, ask, onAction) {
    const vm = askCardViewModel(ask);
    if (vm.resolvable) return renderResolvable(doc, vm, onAction);
    if (vm.samePerson) return renderSamePerson(doc, vm, onAction);
    if (vm.personConfirm) return renderPersonConfirm(doc, vm, onAction);
    const card = el(doc, "div", "rounded-lg border border-sky-800 bg-gray-800/60 p-3 space-y-2");
    card.setAttribute("data-ask-id", vm.askId);
    card.appendChild(el(doc, "div", "text-sm text-gray-100", vm.question));
    // Several pictures: each one after the first gets "is this the same
    // person?" (default yes), so one cluster holding two people can be said.
    const multi = vm.whoIsThis && vm.images.length > 1;
    const choices = {};
    const pictureNames = {};
    if (vm.images.length) {
      card.appendChild(renderImages(doc, vm, !multi ? null : function (i) {
        if (i === 0) return el(doc, "div", "text-[11px] text-gray-400", "The one I am asking about");
        const wrap = el(doc, "div", "space-y-1");
        const id = nextId("ask-pic");
        const label = el(doc, "label", "block text-[11px] text-gray-400", "Picture " + (i + 1) + " is:");
        label.setAttribute("for", id);
        const select = el(doc, "select", "w-full bg-gray-900 text-gray-100 text-xs rounded px-1 py-1 border border-gray-700");
        setStyle(select, { width: "100%", minWidth: "0", boxSizing: "border-box" });
        select.setAttribute("id", id);
        select.setAttribute("data-ask-picture", String(i));
        [["same", "The same person"], ["different", "Someone else"], ["unsure", "Not sure"]].forEach(function (o) {
          const opt = el(doc, "option", "", o[1]);
          opt.setAttribute("value", o[0]);
          select.appendChild(opt);
        });
        select.value = "same";
        const other = nameField(doc, vm.askId, "Who is it? (optional)", "picture-" + i);
        showNode(other.wrap, false);
        choices[i] = "same";
        pictureNames[i] = other.input;
        select.addEventListener("change", function () {
          choices[i] = String(select.value || "same");
          showNode(other.wrap, choices[i] === "different");
        });
        wrap.appendChild(label);
        wrap.appendChild(select);
        wrap.appendChild(other.wrap);
        return wrap;
      }));
    }
    if (vm.askedAt) card.appendChild(el(doc, "div", "text-[11px] text-gray-500", "Asked " + vm.askedAt));
    const row = el(doc, "div", "flex items-center gap-2");
    const input = el(doc, "input", "flex-1 bg-gray-900 text-gray-100 text-xs rounded px-2 py-1 border border-gray-700");
    setStyle(input, { minWidth: "0" });
    input.setAttribute("type", "text");
    input.setAttribute("maxlength", "500");
    input.setAttribute("placeholder", multi ? "Who is the first picture?" : "Your answer");
    input.setAttribute("aria-label", multi ? "Who is the first picture?" : "Your answer");
    input.setAttribute("data-ask-input", vm.askId);
    const answerBtn = el(doc, "button", "text-xs bg-sky-700 hover:bg-sky-600 text-white rounded px-3 py-1", "Answer");
    answerBtn.setAttribute("type", "button");
    answerBtn.setAttribute("data-ask-action", "answer");
    const dismissBtn = el(doc, "button", "text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded px-3 py-1 border border-gray-700", "Dismiss");
    dismissBtn.setAttribute("type", "button");
    dismissBtn.setAttribute("data-ask-action", "dismiss");
    // "I don't know": closes the card without a name (see the header comment).
    let dontKnowBtn = null;
    if (vm.whoIsThis) {
      dontKnowBtn = el(doc, "button", "text-xs bg-gray-800 hover:bg-gray-700 text-gray-200 rounded px-3 py-1 border border-gray-600", "I don't know");
      dontKnowBtn.setAttribute("type", "button");
      dontKnowBtn.setAttribute("data-ask-action", "dont-know");
      dontKnowBtn.setAttribute("title", "No name is saved. Orion may ask again later.");
    }
    const note = el(doc, "div", "text-[11px] text-amber-300 hidden");
    // An empty Answer points at the button that says what she means.
    if (dontKnowBtn) note.setAttribute("data-empty-hint", "Type a name first, or press I don't know.");
    // Every button is locked while a request is in flight, so a double
    // click cannot send a second POST and flash a spurious 409.
    const buttons = dontKnowBtn ? [answerBtn, dontKnowBtn, dismissBtn] : [answerBtn, dismissBtn];
    function run(action, text, extra) {
      buttons.forEach(function (b) { b.disabled = true; });
      return Promise.resolve(onAction(vm.askId, action, text, note, undefined, extra)).finally(function () {
        buttons.forEach(function (b) { b.disabled = false; });
      });
    }
    answerBtn.addEventListener("click", function () {
      if (!multi) return run("answer", input.value);
      const names = {};
      Object.keys(pictureNames).forEach(function (k) { names[k] = pictureNames[k].value; });
      return run("answer", input.value, { pictures: pictureSubmission(vm.images, choices, names) });
    });
    dismissBtn.addEventListener("click", function () {
      return run("dismiss", "");
    });
    if (dontKnowBtn) {
      // Closes the whole card: any per-picture choices are not sent (she does not know who it is).
      dontKnowBtn.addEventListener("click", function () {
        return run("dismiss", "", { dontKnow: true });
      });
    }
    row.appendChild(input);
    row.appendChild(answerBtn);
    if (dontKnowBtn) row.appendChild(dontKnowBtn);
    row.appendChild(dismissBtn);
    card.appendChild(row);
    card.appendChild(note);
    return card;
  }

  // extra: optional {names} / {pictures} sent with an answer.
  async function submitAction(fetchFn, askId, action, answerText, resolution, extra) {
    const url = "/api/asks/" + encodeURIComponent(askId) + "/" + action;
    const init = { method: "POST", headers: { "Content-Type": "application/json" } };
    if (action === "answer") {
      const body = { answer: String(answerText || "").trim() };
      if (extra && extra.names && Object.keys(extra.names).length) body.names = extra.names;
      if (extra && Array.isArray(extra.pictures) && extra.pictures.length) body.pictures = extra.pictures;
      init.body = JSON.stringify(body);
    }
    if (action === "resolve") {
      init.body = JSON.stringify({ resolution: String(resolution || ""), note: String(answerText || "").trim() });
    }
    if (action === "dismiss" && extra && extra.dontKnow) {
      init.body = JSON.stringify({ reason: "dont_know" });
    }
    const resp = await fetchFn(url, init);
    let body = null;
    try {
      body = await resp.json();
    } catch (_e) {
      body = null;
    }
    return { ok: resp.ok, status: resp.status, body: body };
  }

  // At most this many questions show at once; the rest scroll inside the list so
  // a long queue cannot push the chat off the top of the Hub.
  const MAX_VISIBLE_ASKS = 2;

  // Pixel height of the first `max` cards plus the gaps between them, or null
  // when everything already fits (no cap needed).
  function visibleAsksMaxHeight(heights, gap, max) {
    const limit = max == null ? MAX_VISIBLE_ASKS : max;
    if (!heights || heights.length <= limit) return null;
    let total = 0;
    for (let i = 0; i < limit; i++) total += Number(heights[i]) || 0;
    return total + (Number(gap) || 0) * (limit - 1);
  }

  function capVisibleAsks(win, list) {
    const kids = Array.prototype.slice.call(list.children);
    // space-y-* spaces cards with margins, not row-gap.
    const gap = kids[1] ? parseFloat(win.getComputedStyle(kids[1]).marginTop) || 0 : 0;
    const px = visibleAsksMaxHeight(kids.map(function (k) { return k.offsetHeight; }), gap);
    // px of 0 means the Hub tab is hidden (nothing measurable): do not cap.
    list.style.maxHeight = !px ? "" : Math.ceil(px) + "px";
  }

  function mount(doc, fetchFn) {
    const list = doc.getElementById("visionAsksList");
    const status = doc.getElementById("visionAsksStatus");
    if (!list || !status) return null;

    // True while Juniper is mid-answer: a box has focus or unsent text. The
    // timed poll skips then, because refresh() rebuilds every card and would
    // wipe what she is typing.
    function isTyping() {
      const active = doc.activeElement;
      if (active && active.getAttribute && active.getAttribute("data-ask-input")) return true;
      return hasText(list);
    }

    // A picked choice counts too: a rebuild would clear it.
    function hasText(node) {
      if (!node) return false;
      if (node.getAttribute && node.getAttribute("data-ask-input") && String(node.value || "").trim()) return true;
      if (node.getAttribute && node.getAttribute("data-ask-choice") && node.checked) return true;
      if (node.getAttribute && node.getAttribute("data-ask-picture") && node.value && node.value !== "same") return true;
      const kids = node.children || [];
      for (let i = 0; i < kids.length; i++) if (hasText(kids[i])) return true;
      return false;
    }

    async function refresh() {
      try {
        const resp = await fetchFn("/api/asks?status=open");
        if (!resp.ok) {
          status.textContent = errorLine(resp.status);
          return;
        }
        const data = await resp.json();
        const asks = (data && data.asks) || [];
        status.textContent = statusLine(asks);
        while (list.firstChild) list.removeChild(list.firstChild);
        asks.forEach(function (ask) {
          list.appendChild(renderAsk(doc, ask, onAction));
        });
        if (doc.defaultView) capVisibleAsks(doc.defaultView, list);
      } catch (_e) {
        status.textContent = "Can't reach the Hub to load Orion's questions.";
      }
    }

    async function onAction(askId, action, answerText, note, resolution, extra) {
      if (action === "answer" && !String(answerText || "").trim()) {
        note.textContent = (note.getAttribute && note.getAttribute("data-empty-hint")) || "Type an answer first, or press Dismiss.";
        note.classList.remove("hidden");
        return;
      }
      if (action === "resolve" && resolution === "revised" && !String(answerText || "").trim()) {
        note.textContent = errorLine(422, "revised_needs_note");
        note.classList.remove("hidden");
        return;
      }
      try {
        const res = await submitAction(fetchFn, askId, action, answerText, resolution, extra);
        if (!res.ok) {
          note.textContent = errorLine(res.status, res.body && res.body.detail);
          note.classList.remove("hidden");
          if (res.status !== 409 && res.status !== 404) return;
        }
      } catch (_e) {
        note.textContent = "Can't reach the Hub; your answer was not saved.";
        note.classList.remove("hidden");
        return;
      }
      await refresh();
    }

    refresh();
    const timer = global.setInterval(function () {
      if (!doc.hidden && !isTyping()) refresh();
    }, POLL_MS);
    return { refresh: refresh, onAction: onAction, isTyping: isTyping, stop: function () { global.clearInterval(timer); } };
  }

  const api = {
    askImageView: askImageView,
    askImageViews: askImageViews,
    imageAlt: imageAlt,
    isSamePerson: isSamePerson,
    pairIds: pairIds,
    samePersonSubmission: samePersonSubmission,
    isPersonConfirm: isPersonConfirm,
    personConfirmName: personConfirmName,
    personConfirmSubmission: personConfirmSubmission,
    personConfirmPicturesSubmission: personConfirmPicturesSubmission,
    pictureSubmission: pictureSubmission,
    askCardViewModel: askCardViewModel,
    statusLine: statusLine,
    errorLine: errorLine,
    isResolvable: isResolvable,
    revisionProblem: revisionProblem,
    submitAction: submitAction,
    renderAsk: renderAsk,
    mount: mount,
    visibleAsksMaxHeight: visibleAsksMaxHeight,
  };

  global.OrionVisionAsks = api;
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

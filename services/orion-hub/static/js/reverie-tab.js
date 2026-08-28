/**
 * Reverie tab -- historical, human-visible view of both reverie chains.
 * Self-contained module (same lifecycle contract as attention-organ.js):
 * window.OrionReverie = { activate, deactivate, refresh }.
 *
 * No continuous polling -- this is a historical browsing tool, not a live
 * telemetry dashboard (the text chain ticks every ~90s, the visual chain
 * every ~600s; a manual refresh is the honest cadence for "look back at what
 * happened", not a fast poll loop). One fetch on activate(), and whenever
 * the operator clicks Refresh, switches sub-view, or pages.
 *
 * Backed by services/orion-hub/scripts/reverie_routes.py
 * (/api/reverie/visual/recent, /api/reverie/visual/image/{sha256},
 * /api/reverie/text/recent).
 *
 * Visual cockpit layout (this file's real subject): a static pipeline
 * diagram (PIPELINE_STAGES below, one node per real function this chain
 * actually calls -- see services/orion-thought/app/visual_chain.py, the
 * ground truth this diagram is drawn from, not an idealized version of it),
 * then one detail card per run showing the exact prompt used, the image it
 * produced, the caption/description that came back, and this run's real
 * egress -- what happens to the output, including the honest disclosure
 * that nothing consumes it further today (design doc §8, Patch 3
 * territory). Real offset-based pagination, not a single fixed-limit fetch.
 */
(function () {
  let active = false;
  let subview = "visual"; // "visual" | "text"
  let loaded = { visual: false, text: false };
  const VISUAL_PAGE_SIZE = 9;
  let diagramRendered = false;
  // Cursor-based paging (review finding: OFFSET has no stable meaning
  // against a table that gets a new row every ~600s from the live worker --
  // a concurrent insert shifts every row's offset mid-session). cursorStack
  // holds the `before` value used to fetch each page already visited --
  // index 0 is always null (first/newest page) -- so "← Newer" replays an
  // already-known cursor instead of re-deriving one, and "Older →" only
  // ever needs the cursor the server just handed back (`next_before`).
  let cursorStack = [null];
  let pageIndex = 0;
  let visualHasMore = false;
  // Out-of-order fetch guard (review finding): a double Prev/Next click or a
  // Refresh mid-flight can let an older request's response resolve after a
  // newer one already rendered. Only the response matching the most recently
  // issued request is allowed to touch the DOM.
  let visualRequestSeq = 0;

  function el(id) {
    return document.getElementById(id);
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString();
    } catch (e) {
      return iso;
    }
  }

  function setStatus(msg) {
    const s = el("reverieStatus");
    if (s) s.textContent = msg;
  }

  // --- Pipeline diagram ---------------------------------------------------
  //
  // One node per real call this chain makes, in call order. `file` is a
  // repo-relative path:function pointer so this diagram stays inspectable
  // evidence (CLAUDE.md §0A "if Orion says it reasoned... there must be
  // inspectable evidence") rather than a decorative illustration -- click a
  // node to see exactly which function produced that stage.

  const PIPELINE_STAGES = [
    {
      id: "input",
      title: "1 · Continuity + context inputs",
      desc:
        "The previous run's own caption (reverie_visual_chain.prior_description), Orion's own " +
        "most recent real reverie-thought interpretation (context_text), a real quantified " +
        "self-study observation (self_study_text), a real shared-life memory (memory_text), " +
        "and how many consecutive runs have used continuity so far (continuity_streak).",
      detail:
        'Fixed seed only when NONE of prior_description/context_text/self_study_text/memory_text ' +
        'exist (fresh install, no history yet): "a calm orion, soft abstract light, dreaming". ' +
        "Every later run reads all five from Postgres, concurrently.",
      file: "services/orion-thought/app/store.py :: load_latest_visual_chain_continuity_state, load_latest_reverie_interpretation, load_latest_self_study_reflection, load_latest_memory_crystallization",
    },
    {
      id: "continuity-cap",
      title: "2 · Continuity cap check",
      desc:
        "After ORION_VISUAL_CHAIN_CONTINUITY_MAX_RUNS (default 3) consecutive runs carrying " +
        "continuity forward, THIS run forces continuity to drop from its own prompt.",
      detail:
        "Patch 4 (shipped 2026-08-27): live-caught the same day Juniper reported identical " +
        "\"Roman aqueduct\" imagery unbroken for 10+ runs -- a short context_text clause has " +
        "nowhere near the prompt weight of a long, concrete continuity description, so context-" +
        "seeding alone never actually redirected the diffusion model. This is a mechanical " +
        "guarantee instead of a prompt-reweighting guess: no off switch by design (0 resets " +
        "every run). A reset run still seeds from context_text when real narration exists.",
      file: "services/orion-thought/app/visual_chain.py :: resolve_visual_chain_continuity",
    },
    {
      id: "prompt",
      title: "3 · Prompt construction",
      desc:
        "The (possibly reset) continuity input and all THREE context-seeds are blended into " +
        "one prompt text -- continuity keeps the image chain visually coherent frame-to-frame " +
        "when allowed to run, the context-seeds keep it grounded in what Orion is actually " +
        "narrating, observing, and remembering.",
      detail:
        "Patch 3 (shipped): a deliberately narrow first context-seed slice -- Orion's own " +
        "reverie-thought interpretation, already surfaced by this tab's Text sub-view (no new " +
        "privacy surface). Patch 5 (shipped): a second, richer context-seed from the self-study " +
        "analysis system's real quantified self-observation. Patch 6 (shipped): a third, from " +
        "the Recall system's memory_crystallizations table -- real shared-life content, " +
        "unfiltered by content (this route has no external audience beyond its one viewer, " +
        "who is also that content's original source). Falls back to the fixed seed string only " +
        "when all four inputs are empty.",
      file: "services/orion-thought/app/visual_chain.py :: build_visual_prompt",
    },
    {
      id: "generate",
      title: "4 · Generate",
      desc: "POST {prompt} to orion-diffusion-host's /generate -- returns raw PNG bytes.",
      detail:
        "A non-2xx response (including diffusion-host's documented 429 busy-reject) or a network " +
        "failure ends the run here: terminal_reason=generation_failed, no image, no artifact row.",
      file: "services/orion-thought/app/visual_chain.py :: call_diffusion_generate",
    },
    {
      id: "store",
      title: "5 · Store + upload (parallel)",
      desc:
        "The same PNG bytes are content-addressed to local disk AND uploaded to " +
        "orion-percept-store (hash-verified both ways) at the same time.",
      detail:
        "A store failure is a real generation failure (nothing to persist). An upload failure only " +
        "degrades the next stage -- the image is still kept and gets an artifact row with no caption.",
      file: "services/orion-thought/app/visual_chain.py :: run_visual_chain_once",
    },
    {
      id: "observe",
      title: "6 · Observe (caption)",
      desc:
        "RPC to orion-vision-host's existing caption_frame task, over the shared bus request/reply " +
        "channel -- the same captioner used elsewhere in the system, not a dedicated model.",
      detail:
        "Best-effort: timeout, decode failure, ok=false, or an empty caption all return no caption " +
        "rather than raising or fabricating one. A failed observation still keeps the image.",
      file: "services/orion-thought/app/visual_chain.py :: request_caption",
    },
    {
      id: "persist",
      title: "7 · Persist",
      desc: "reverie_visual_chain (this run) and reverie_visual_artifact (the image) rows are written.",
      detail:
        "Only a real, non-empty caption advances continuity -- a failed observation carries the " +
        "*previous* run's prior_description forward unchanged instead of losing the thread. " +
        "continuity_streak/continuity_reset are recorded here too, on both this path and " +
        "generation_failed, so a failed run still records the correct streak for whichever run " +
        "next picks continuity back up.",
      file: "services/orion-thought/app/store.py :: persist_reverie_visual_chain / persist_reverie_visual_artifact",
    },
    {
      id: "loop",
      title: "↺ Loops back to 1",
      desc: "This run's caption becomes the next run's continuity input, ~600s later.",
      detail: "ORION_VISUAL_CHAIN_INTERVAL_SEC controls the cadence (default 600s).",
      file: "services/orion-thought/app/visual_chain.py :: run_visual_chain_worker",
    },
  ];

  const EGRESS_NODE = {
    id: "egress",
    title: "Egress",
    desc:
      "Nothing else consumes this today. No bus publish about the run, no downstream trigger. " +
      "Honest dead end, not a hidden one -- Patch 3 territory.",
    detail:
      "Design spec §8 non-goals. The image is real (stored + captioned) but the only thing reading " +
      "it back is this same loop's own next prompt, and this Hub tab.",
    file: "docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md",
  };

  function stageNodeHtml(stage, extraClass) {
    return `
      <button type="button" class="reverie-diagram-node text-left rounded-lg border border-gray-700 bg-gray-950/60 px-3 py-2 hover:border-indigo-500 transition-colors ${extraClass || ""}"
              data-stage="${stage.id}">
        <div class="text-[11px] font-semibold text-gray-200">${escapeHtml(stage.title)}</div>
        <div class="text-[11px] text-gray-500 mt-1 leading-snug">${escapeHtml(stage.desc)}</div>
      </button>`;
  }

  function renderPipelineDiagram() {
    const container = el("reverieVisualDiagram");
    if (!container) return;
    const arrow = `<div class="text-gray-600 text-lg leading-none px-1 select-none" aria-hidden="true">→</div>`;
    const chainNodes = PIPELINE_STAGES.map((s) => stageNodeHtml(s)).join(arrow);
    container.innerHTML = `
      <div class="rounded-xl border border-gray-800 bg-gray-950/30 p-3">
        <div class="text-xs text-gray-400 mb-2">
          Generate → store → observe → interpret, one real function call per stage. Click a node for the exact code it runs.
        </div>
        <div class="flex flex-wrap items-stretch gap-1">${chainNodes}</div>
        <div class="flex items-center gap-1 mt-2">
          <div class="text-gray-600 text-lg leading-none px-1 select-none" aria-hidden="true">⇥</div>
          ${stageNodeHtml(EGRESS_NODE, "border-amber-800/60")}
        </div>
        <div id="reverieDiagramDetail" class="text-xs text-gray-400 mt-3 min-h-[2.5rem] border-t border-gray-800 pt-2">
          Click any stage above for what it actually does and the exact function it runs.
        </div>
      </div>`;

    const detailBox = el("reverieDiagramDetail");
    container.querySelectorAll(".reverie-diagram-node").forEach((btn) => {
      btn.addEventListener("click", () => {
        const stage =
          PIPELINE_STAGES.find((s) => s.id === btn.dataset.stage) ||
          (EGRESS_NODE.id === btn.dataset.stage ? EGRESS_NODE : null);
        if (!stage || !detailBox) return;
        detailBox.innerHTML = `
          <div class="text-gray-200 font-semibold mb-1">${escapeHtml(stage.title)}</div>
          <div class="mb-1">${escapeHtml(stage.detail)}</div>
          <code class="text-gray-500">${escapeHtml(stage.file)}</code>`;
      });
    });
  }

  // --- Visual panel ---------------------------------------------------------

  function renderVisualChain(chain) {
    const artifact = (chain.artifacts && chain.artifacts[0]) || null;

    if (chain.error) {
      return `
        <div class="rounded-xl border border-red-900/60 bg-red-950/10 p-3 flex flex-col gap-2">
          <div class="text-xs text-red-400 font-semibold">Generation failed -- no image produced</div>
          <p class="text-xs text-gray-400">${escapeHtml(chain.error)}</p>
          ${chain.prompt ? `<p class="text-xs text-gray-500">attempted prompt: <span class="text-gray-400">${escapeHtml(chain.prompt)}</span></p>` : ""}
          <div class="flex justify-between items-center mt-1 text-[11px] text-gray-500">
            <span>${fmtTime(chain.created_at)}</span>
            <span class="px-2 py-0.5 rounded-full border border-red-800 text-red-400">${escapeHtml(chain.terminal_reason)}</span>
          </div>
        </div>`;
    }

    const img = artifact
      ? `<img src="${artifact.image_url}" loading="lazy" class="rounded-lg w-full object-cover" style="max-height: 280px;" alt="reverie image" />`
      : `<div class="text-xs text-gray-500 italic p-4">no image stored for this chain</div>`;
    const caption = artifact && artifact.description
      ? `<p class="text-sm text-gray-200 mt-2">“${escapeHtml(artifact.description)}”</p>`
      : `<p class="text-xs text-gray-500 italic mt-2">not captioned (re-observation failed or was rejected -- honest null, not fabricated)</p>`;
    const contextBlock = chain.context_text
      ? `<div class="mt-2 rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
           <div class="text-[10px] uppercase tracking-wide text-gray-600">Context-seed (Orion's own reverie thought)</div>
           <div class="text-xs text-gray-400 mt-0.5">${escapeHtml(chain.context_text)}</div>
         </div>`
      : "";
    const selfStudyBlock = chain.self_study_text
      ? `<div class="mt-2 rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
           <div class="text-[10px] uppercase tracking-wide text-gray-600">Self-study observation (real quantified finding)</div>
           <div class="text-xs text-gray-400 mt-0.5">${escapeHtml(chain.self_study_text)}</div>
         </div>`
      : "";
    const memoryBlock = chain.memory_text
      ? `<div class="mt-2 rounded border border-gray-800 bg-gray-950/40 px-2 py-1.5">
           <div class="text-[10px] uppercase tracking-wide text-gray-600">Memory (shared-life crystallization)</div>
           <div class="text-xs text-gray-400 mt-0.5">${escapeHtml(chain.memory_text)}</div>
         </div>`
      : "";
    const continuityLabel = chain.continuity_reset
      ? "Prompt used (continuity RESET this run -- seeded fresh from the context-seed above)"
      : `Prompt used (blends the prior caption with the context-seed above` +
        (typeof chain.continuity_streak === "number"
          ? `; continuity streak ${chain.continuity_streak}`
          : "") +
        `)`;
    const promptBlock = chain.prompt
      ? `<div class="mt-2 rounded border ${chain.continuity_reset ? "border-amber-800/60" : "border-gray-800"} bg-gray-950/40 px-2 py-1.5">
           <div class="text-[10px] uppercase tracking-wide ${chain.continuity_reset ? "text-amber-600" : "text-gray-600"}">${escapeHtml(continuityLabel)}</div>
           <div class="text-xs text-gray-400 mt-0.5">${escapeHtml(chain.prompt)}</div>
         </div>`
      : "";
    const egressLine = artifact && artifact.description
      ? `stored on disk + orion-percept-store, captioned via orion-vision-host, this caption becomes the <em>next</em> run's prompt input. Nothing else reads it today.`
      : artifact
        ? `stored on disk + orion-percept-store; captioning failed so the previous run's caption carries forward instead. Nothing else reads this image today.`
        : `no image was produced this run.`;

    return `
      <div class="rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-1">
        ${img}
        ${caption}
        ${contextBlock}
        ${selfStudyBlock}
        ${memoryBlock}
        ${promptBlock}
        <div class="text-[11px] text-gray-600 mt-1">egress: ${egressLine}</div>
        <div class="flex justify-between items-center mt-2 text-[11px] text-gray-500">
          <span>${fmtTime(chain.created_at)}</span>
          <span class="px-2 py-0.5 rounded-full border border-gray-700">${escapeHtml(chain.terminal_reason)}</span>
        </div>
      </div>`;
  }

  function renderVisualPager() {
    const pager = el("reverieVisualPager");
    if (!pager) return;
    const hasPrev = pageIndex > 0;
    const hasNext = visualHasMore;
    pager.innerHTML = `
      <span>Page ${pageIndex + 1}</span>
      <span class="flex gap-2">
        <button type="button" id="reveriePagerPrev" class="px-2 py-1 rounded border border-gray-700 ${hasPrev ? "hover:bg-gray-800 text-gray-300" : "opacity-40 cursor-not-allowed text-gray-600"}" ${hasPrev ? "" : "disabled"}>← Newer</button>
        <button type="button" id="reveriePagerNext" class="px-2 py-1 rounded border border-gray-700 ${hasNext ? "hover:bg-gray-800 text-gray-300" : "opacity-40 cursor-not-allowed text-gray-600"}" ${hasNext ? "" : "disabled"}>Older →</button>
      </span>`;
    const prevBtn = el("reveriePagerPrev");
    const nextBtn = el("reveriePagerNext");
    if (prevBtn && hasPrev) {
      prevBtn.addEventListener("click", () => {
        pageIndex = Math.max(0, pageIndex - 1);
        loadVisual();
      });
    }
    if (nextBtn && hasNext) {
      nextBtn.addEventListener("click", () => {
        // Only ever pages forward into a cursor the server itself just
        // handed back in the current page's response (pushed in loadVisual
        // below) -- never derived/guessed client-side.
        pageIndex = pageIndex + 1;
        loadVisual();
      });
    }
  }

  async function loadVisual() {
    const container = el("reverieVisualGrid");
    if (!container) return;
    if (!diagramRendered) {
      renderPipelineDiagram();
      diagramRendered = true;
    }
    setStatus("Loading visual reverie chain…");
    const seq = ++visualRequestSeq;
    const before = cursorStack[pageIndex];
    try {
      const url = before
        ? `/api/reverie/visual/recent?limit=${VISUAL_PAGE_SIZE}&before=${encodeURIComponent(before)}`
        : `/api/reverie/visual/recent?limit=${VISUAL_PAGE_SIZE}`;
      const resp = await fetch(url);
      const data = await resp.json();
      if (seq !== visualRequestSeq) return; // a newer request already won -- discard this stale response
      if (!data.ok) throw new Error("bad response");
      if (!data.chains.length) {
        container.innerHTML = pageIndex === 0
          ? `<div class="text-sm text-gray-500 italic p-4">No visual reverie chains yet -- ORION_VISUAL_CHAIN_ENABLED may be off, or none have run.</div>`
          : `<div class="text-sm text-gray-500 italic p-4">No more runs on this page.</div>`;
      } else {
        container.innerHTML = data.chains.map(renderVisualChain).join("");
      }
      visualHasMore = !!data.has_more;
      if (data.next_before && cursorStack[pageIndex + 1] === undefined) {
        cursorStack[pageIndex + 1] = data.next_before;
      }
      renderVisualPager();
      loaded.visual = true;
      setStatus(`Loaded ${data.chains.length} visual chain(s) (page ${pageIndex + 1}).`);
    } catch (e) {
      if (seq !== visualRequestSeq) return;
      setStatus("Failed to load visual reverie chain: " + e.message);
    }
  }

  // --- Text panel ---------------------------------------------------------

  function renderTextChain(chain) {
    const thoughts = (chain.thoughts || [])
      .map(
        (t) => `
        <div class="border-l-2 border-gray-700 pl-3 py-1">
          <p class="text-sm text-gray-200">${escapeHtml(t.interpretation)}</p>
          <p class="text-[11px] text-gray-500">${fmtTime(t.created_at)} · salience ${Number(t.salience).toFixed(2)}</p>
        </div>`
      )
      .join("");
    const badges = [];
    if (chain.downstream && chain.downstream.compaction_queued) {
      badges.push(
        `<span class="px-2 py-0.5 rounded-full border border-amber-700 text-amber-400 text-[11px]">queued for dream compaction</span>`
      );
    }
    if (chain.downstream && chain.downstream.theme_resonance_alert_count > 0) {
      badges.push(
        `<span class="px-2 py-0.5 rounded-full border border-red-700 text-red-400 text-[11px]">${chain.downstream.theme_resonance_alert_count} resonance alert(s) on this theme</span>`
      );
    }
    return `
      <div class="rounded-xl border border-gray-800 bg-gray-950/40 p-3 flex flex-col gap-2">
        <div class="flex justify-between items-center text-[11px] text-gray-500">
          <span>${fmtTime(chain.created_at)} · theme <span class="text-gray-400">${escapeHtml(chain.theme_key || "unknown")}</span></span>
          <span class="px-2 py-0.5 rounded-full border border-gray-700">${escapeHtml(chain.terminal_reason)}</span>
        </div>
        <div class="flex flex-col gap-1">${thoughts || '<span class="text-xs text-gray-500 italic">no thoughts recorded</span>'}</div>
        <div class="flex gap-2 flex-wrap">${badges.join("") || '<span class="text-[11px] text-gray-600 italic">no downstream effect yet</span>'}</div>
      </div>`;
  }

  async function loadText() {
    const container = el("reverieTextGrid");
    if (!container) return;
    setStatus("Loading text reverie chain…");
    try {
      const resp = await fetch("/api/reverie/text/recent?limit=30");
      const data = await resp.json();
      if (!data.ok) throw new Error("bad response");
      if (!data.chains.length) {
        container.innerHTML = `<div class="text-sm text-gray-500 italic p-4">No text reverie chains yet.</div>`;
      } else {
        container.innerHTML = data.chains.map(renderTextChain).join("");
      }
      loaded.text = true;
      setStatus(`Loaded ${data.chains.length} text chain(s).`);
    } catch (e) {
      setStatus("Failed to load text reverie chain: " + e.message);
    }
  }

  // --- Sub-view switching ---------------------------------------------------

  function showSubview(name) {
    subview = name;
    const visualPanel = el("reverieVisualPanel");
    const textPanel = el("reverieTextPanel");
    const visualBtn = el("reverieSubtabVisual");
    const textBtn = el("reverieSubtabText");
    if (visualPanel) visualPanel.classList.toggle("hidden", subview !== "visual");
    if (textPanel) textPanel.classList.toggle("hidden", subview !== "text");
    if (visualBtn) visualBtn.classList.toggle("bg-gray-700", subview === "visual");
    if (textBtn) textBtn.classList.toggle("bg-gray-700", subview === "text");
    if (subview === "visual" && !loaded.visual) loadVisual();
    if (subview === "text" && !loaded.text) loadText();
  }

  function refresh() {
    if (subview === "visual") {
      cursorStack = [null];
      pageIndex = 0;
      loaded.visual = false;
      loadVisual();
    } else {
      loaded.text = false;
      loadText();
    }
  }

  let wired = false;

  function wireOnce() {
    // Idempotency guard (review finding): activate() also calls this, and a
    // tab can be activated many times in one session -- without the guard,
    // click listeners accumulate unboundedly and a single click fires the
    // handler once per prior activation.
    if (wired) return;
    wired = true;
    const visualBtn = el("reverieSubtabVisual");
    const textBtn = el("reverieSubtabText");
    const refreshBtn = el("reverieRefreshBtn");
    if (visualBtn) visualBtn.addEventListener("click", () => showSubview("visual"));
    if (textBtn) textBtn.addEventListener("click", () => showSubview("text"));
    if (refreshBtn) refreshBtn.addEventListener("click", refresh);
  }

  function activate() {
    active = true;
    wireOnce();
    if (subview === "visual" && !loaded.visual) loadVisual();
    if (subview === "text" && !loaded.text) loadText();
  }

  function deactivate() {
    active = false;
    // Rendered DOM stays in place -- same contract as Attention Organ.
  }

  document.addEventListener("DOMContentLoaded", wireOnce);

  window.OrionReverie = { activate, deactivate, refresh };
})();

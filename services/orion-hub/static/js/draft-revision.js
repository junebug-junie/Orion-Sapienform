(function (global) {
  // Draft-first display for unified chat turns (spec L8, 2026-10-06).
  //
  // The Hub sends a `draft_preview` frame as soon as the reply writer's draft
  // exists, then the usual `final` frame once the finalize judge is done. This
  // module owns only the DOM swap: show the draft as a provisional Orion
  // message, then replace it in place with the real final message (built by
  // app.js::appendMessage, so it carries all the usual chrome), marked
  // "revised" with the reason when the judge changed the text.
  //
  // Kept standalone (like turn-timer.js) so the swap is unit-testable under
  // `node --test` without the 600KB app.js. Uses only createElement,
  // appendChild, insertBefore, removeChild, children, dataset, className and
  // textContent, so a tiny fake DOM is enough to exercise it.

  const DRAFT_ATTR_KEY = 'draftPreviewFor';

  const REASON_TEXT = {
    strain_unresolved: 'the check found an unresolved strain',
    misaligned: 'the check found it out of line with the stance',
    uncertain: 'the check was unsure about it',
    finalize_changed: 'the final pass changed it',
  };

  function revisedLabel(reason) {
    const key = String(reason || '').trim();
    const why = REASON_TEXT[key] || key || 'the check changed it';
    return `revised: ${why}`;
  }

  function childList(container) {
    return container && container.children ? Array.from(container.children) : [];
  }

  function findDraft(container, correlationId) {
    const corr = String(correlationId || '').trim();
    if (!corr) return null;
    for (const el of childList(container)) {
      if (el && el.dataset && el.dataset[DRAFT_ATTR_KEY] === corr) return el;
    }
    return null;
  }

  function showDraft(container, frame, opts) {
    if (!container || !frame) return null;
    const corr = String(frame.correlation_id || '').trim();
    const text = typeof frame.draft_text === 'string' ? frame.draft_text : '';
    if (!corr || !text.trim()) return null;
    const doc = (opts && opts.document) || global.document;
    // One draft per turn: a repeated frame for the same turn is ignored.
    const existing = findDraft(container, corr);
    if (existing) return existing;

    const div = doc.createElement('div');
    div.dataset[DRAFT_ATTR_KEY] = corr;
    div.dataset.draftState = 'checking';
    div.dataset.role = 'assistant';
    div.dataset.sender = 'Orion';
    div.className = 'mb-2 border-b border-gray-800/50 pb-2 last:border-0 om-draft-preview';

    const headerRow = doc.createElement('div');
    headerRow.className = 'mb-1 flex items-center gap-3';
    const header = doc.createElement('p');
    header.className = 'font-bold text-green-300';
    header.textContent = 'Orion';
    const badge = doc.createElement('span');
    badge.className = 'om-draft-badge text-[10px] text-gray-400';
    badge.dataset.draftBadge = '1';
    badge.textContent = 'draft, still being checked';
    headerRow.appendChild(header);
    headerRow.appendChild(badge);

    const body = doc.createElement('p');
    body.dataset.messageBody = '1';
    const rendered = opts && typeof opts.renderMarkdown === 'function' ? opts.renderMarkdown(text) : null;
    if (rendered) {
      body.className = 'text-white om-md';
      body.appendChild(rendered);
    } else {
      body.className = 'text-white whitespace-pre-wrap';
      body.textContent = text;
    }
    div.appendChild(headerRow);
    div.appendChild(body);
    container.appendChild(div);
    return div;
  }

  function buildRevisedMarker(doc, reason) {
    const marker = doc.createElement('p');
    marker.className = 'om-revised-marker text-[11px] text-amber-300';
    marker.dataset.revisedMarker = '1';
    marker.textContent = revisedLabel(reason);
    return marker;
  }

  // Replace the draft with the final message node, in the draft's position.
  // Returns 'replaced' | 'revised' | 'no_draft'.
  function settleWithFinal(container, draftNode, finalNode, frame, opts) {
    if (!container || !draftNode) return 'no_draft';
    const doc = (opts && opts.document) || global.document;
    if (finalNode) {
      if (finalNode.parentNode === container || finalNode.parentElement === container) {
        container.removeChild(finalNode);
      }
      container.insertBefore(finalNode, draftNode);
    }
    container.removeChild(draftNode);
    if (finalNode && frame && frame.revised) {
      const marker = buildRevisedMarker(doc, frame.revised_reason);
      const kids = childList(finalNode);
      // Directly under the header row, so it is read before the new text.
      finalNode.insertBefore(marker, kids.length > 1 ? kids[1] : null);
      finalNode.dataset.revised = '1';
      return 'revised';
    }
    return 'replaced';
  }

  // The turn ended without a final (error/deferral): keep the draft visible
  // but say plainly that it was never checked.
  function settleWithoutFinal(draftNode) {
    if (!draftNode) return false;
    draftNode.dataset.draftState = 'unchecked';
    for (const row of childList(draftNode)) {
      for (const el of childList(row)) {
        if (el.dataset && el.dataset.draftBadge) {
          el.textContent = 'draft, not checked (the turn failed before the check finished)';
          el.className = 'om-draft-badge text-[10px] text-yellow-300';
        }
      }
    }
    return true;
  }

  const api = { revisedLabel, findDraft, showDraft, settleWithFinal, settleWithoutFinal };

  global.OrionDraftRevision = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

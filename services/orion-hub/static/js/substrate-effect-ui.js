// services/orion-hub/static/js/substrate-effect-ui.js
// Substrate Effect UI — chip + modal.
// Owns nothing about appraiser semantics. Renders SubstrateEffectViewV1 as-is.

(function () {
  const API_BASE = (window.ORION_HUB_API_BASE || window.location.origin).replace(/\/+$/, '');

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'dataset') Object.assign(node.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    for (const child of [].concat(children)) {
      if (child == null) continue;
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    }
    return node;
  }

  function renderSubstrateEffectChip(summary, { onClick } = {}) {
    if (!summary || typeof summary !== 'object') return null;
    const level = String(summary.level_label || 'NONE').toUpperCase();
    const text = summary.chip_label || 'Substrate Effect';
    const chip = el('button', {
      type: 'button',
      class: 'substrate-effect-chip',
      dataset: { level, turnId: summary.turn_id || '' },
    }, [`Substrate Effect: ${text}`]);
    chip.addEventListener('click', () => {
      const turnId = summary.turn_id;
      if (!turnId) return;
      if (typeof onClick === 'function') onClick(turnId);
      else openSubstrateEffectModal(turnId);
    });
    return chip;
  }

  async function openSubstrateEffectModal(turnId) {
    if (!turnId) return;
    let view;
    try {
      const resp = await fetch(`${API_BASE}/api/chat/turn/${encodeURIComponent(turnId)}/substrate-effect`);
      if (!resp.ok) {
        view = { _error: `${resp.status} ${resp.statusText}` };
      } else {
        view = await resp.json();
      }
    } catch (err) {
      view = { _error: String((err && err.message) || err) };
    }
    document.body.appendChild(renderSubstrateEffectModal(view));
  }

  function renderSubstrateEffectModal(view) {
    const backdrop = el('div', { class: 'substrate-effect-modal-backdrop' });
    const modal = el('div', { class: 'substrate-effect-modal', role: 'dialog' });
    const close = () => {
      backdrop.remove();
      modal.remove();
      document.removeEventListener('keydown', onKey);
    };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    backdrop.addEventListener('click', close);

    const closeBtn = el('button', { type: 'button', class: 'text-gray-400 hover:text-white text-xs' }, ['Close']);
    closeBtn.addEventListener('click', close);
    modal.appendChild(el('header', {}, [el('h2', {}, ['Substrate Effect for This Turn']), closeBtn]));

    const body = el('div', { class: 'body' });
    if (view && view._error) {
      body.appendChild(el('div', { class: 'substrate-effect-section' }, [
        el('h3', {}, ['Error']),
        el('p', { class: 'lede' }, [`Failed to load substrate effect view: ${view._error}`]),
      ]));
    } else {
      body.appendChild(renderOutcome(view));
      if (view.why) body.appendChild(renderWhy(view));
      if (view.behavior_delta) body.appendChild(renderBehaviorDelta(view.behavior_delta));
      if (view.causal_chain && view.causal_chain.length) body.appendChild(renderCausalChain(view.causal_chain));
      if (view.evidence_cards && view.evidence_cards.length) body.appendChild(renderEvidenceCards(view.evidence_cards));
      if (view.scorecard) body.appendChild(renderScorecard(view.scorecard));
      if (view.molecule_summaries && view.molecule_summaries.length) body.appendChild(renderMoleculeSummaries(view.molecule_summaries));
      if (view.raw_debug) body.appendChild(renderRawDebug(view.raw_debug));
    }
    modal.appendChild(body);

    const root = el('div', {});
    root.appendChild(backdrop);
    root.appendChild(modal);
    return root;
  }

  function renderOutcome(view) {
    const o = view.outcome || {};
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Outcome']));
    section.appendChild(el('p', { class: 'lede' }, [o.summary || '']));
    section.appendChild(el('p', { class: 'secondary' }, [
      `Level: ${Number(o.level || 0).toFixed(2)} (${o.level_label || ''}) · Confidence: ${Number(o.confidence || 0).toFixed(2)} (${o.confidence_label || ''})`,
    ]));
    return section;
  }

  function renderWhy(view) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Why']));
    section.appendChild(el('p', { class: 'lede' }, [view.why]));
    return section;
  }

  function renderBehaviorDelta(delta) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['What changed']));
    const before = delta.contract_before || '—';
    const after = delta.contract_after || '—';
    const lede = delta.changed
      ? `Before: ${before} · After: ${after}`
      : `Before: ${before} · After: ${after} — no contract change.`;
    section.appendChild(el('p', { class: 'lede' }, [lede]));
    if (delta.explanation) {
      section.appendChild(el('p', { class: 'secondary' }, [delta.explanation]));
    }
    if (delta.rules_activated && delta.rules_activated.length) {
      const list = el('ul', { class: 'list-disc list-inside text-xs text-gray-200' });
      delta.rules_activated.forEach((rule) => list.appendChild(el('li', {}, [String(rule)])));
      section.appendChild(list);
    }
    return section;
  }

  function renderCausalChain(steps) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Causal chain']));
    const list = el('ol', { class: 'substrate-effect-chain' });
    steps.forEach((step) => {
      list.appendChild(el('li', {}, [
        el('strong', {}, [`${step.index}. ${step.title}`]),
        el('span', { class: 'desc' }, [step.description || '']),
      ]));
    });
    section.appendChild(list);
    return section;
  }

  function renderEvidenceCards(cards) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Evidence']));
    const wrap = el('div', { class: 'grid grid-cols-1 md:grid-cols-2 gap-2' });
    cards.forEach((card) => {
      const node = el('div', { class: 'substrate-effect-card' }, [
        el('h4', {}, [card.label]),
        card.source_span
          ? el('p', { class: 'span' }, [`"${card.source_span}"`])
          : null,
        el('p', { class: 'meta' }, [`Score ${Number(card.score).toFixed(2)} · Confidence ${Number(card.confidence).toFixed(2)}`]),
        card.meaning ? el('p', {}, [card.meaning]) : null,
      ]);
      wrap.appendChild(node);
    });
    section.appendChild(wrap);
    return section;
  }

  function renderScorecard(scorecard) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, [scorecard.title || 'Scorecard']));
    scorecard.items.forEach((item) => {
      const row = el('div', { class: 'substrate-effect-bar' });
      row.appendChild(el('span', {}, [item.label]));
      const track = el('div', { class: 'track' });
      const fill = el('div', { class: 'fill', style: `width:${Math.round(Math.max(0, Math.min(1, item.value)) * 100)}%` });
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('span', { class: 'secondary' }, [Number(item.value).toFixed(2)]));
      section.appendChild(row);
    });
    if (scorecard.final_label) section.appendChild(el('p', { class: 'lede' }, [scorecard.final_label]));
    if (scorecard.explanation) section.appendChild(el('p', { class: 'secondary' }, [scorecard.explanation]));
    return section;
  }

  function renderMoleculeSummaries(summaries) {
    const section = el('section', { class: 'substrate-effect-section' });
    section.appendChild(el('h3', {}, ['Molecules used']));
    summaries.forEach((mol) => {
      section.appendChild(el('div', { class: 'substrate-effect-card' }, [
        el('h4', {}, [`${mol.label}`]),
        el('p', {}, [mol.explanation || '']),
        el('p', { class: 'meta' }, [mol.molecule_id]),
      ]));
    });
    return section;
  }

  function renderRawDebug(rawDebug) {
    const section = el('details', { class: 'substrate-effect-raw' });
    section.appendChild(el('summary', {}, ['Developer payload']));
    section.appendChild(el('pre', {}, [JSON.stringify(rawDebug, null, 2)]));
    return section;
  }

  async function loadRecentEffects(container) {
    try {
      const resp = await fetch(`${API_BASE}/api/substrate-effect/recent?limit=25`);
      if (!resp.ok) {
        container.textContent = `Failed to load recent effects: ${resp.status} ${resp.statusText}`;
        return;
      }
      const data = await resp.json();
      const rows = (data && data.rows) || [];
      container.innerHTML = '';
      if (!rows.length) {
        container.appendChild(el('p', { class: 'text-gray-400' }, ['No substrate effects recorded yet.']));
        return;
      }
      const list = el('div', { class: 'grid grid-cols-1 gap-2' });
      rows.forEach((row) => {
        const card = el('button', {
          type: 'button',
          class: 'substrate-effect-card text-left',
        }, [
          el('h4', {}, [row.chip_label]),
          el('p', { class: 'meta' }, [`${row.stored_at} · ${row.turn_summary}`]),
        ]);
        card.addEventListener('click', () => openSubstrateEffectModal(row.turn_id));
        list.appendChild(card);
      });
      container.appendChild(list);
    } catch (err) {
      container.textContent = `Failed to load recent effects: ${String((err && err.message) || err)}`;
    }
  }

  function initSubstrateEffectTab() {
    const body = document.getElementById('substrateEffectRecentBody');
    const refresh = document.getElementById('substrateEffectRecentRefresh');
    if (!body) return;
    loadRecentEffects(body);
    if (refresh) refresh.addEventListener('click', () => loadRecentEffects(body));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSubstrateEffectTab);
  } else {
    initSubstrateEffectTab();
  }

  window.SubstrateEffectUI = {
    renderChip: renderSubstrateEffectChip,
    openModal: openSubstrateEffectModal,
    renderModal: renderSubstrateEffectModal,
  };
})();

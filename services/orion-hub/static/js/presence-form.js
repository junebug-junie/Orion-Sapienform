(function (global) {
  // Presence modal companion rows ("People with you"), as pure functions so
  // the markup and read-back are unit-testable without a DOM (app.js wires
  // them to #presenceCompanionRows).
  //
  // Each field has a visible label and the three fixed-value fields are
  // dropdowns. The old row was five unlabeled text boxes whose pre-filled
  // values ("other", "nearby", "unknown") hid the placeholders, so a name
  // typed into the wrong box ("Pip" as relationship and age, 2026-10-14,
  // corr 21c8a610) saved fine and then broke presence on the next turn.
  // Values must match PresenceCompanionV1 in orion/schemas/situation.py.
  const CHOICES = {
    relationship: [['child', 'Child'], ['spouse', 'Spouse / partner'], ['friend', 'Friend'], ['coworker', 'Coworker'], ['guest', 'Guest'], ['other', 'Other']],
    age_band: [['child', 'Child'], ['teen', 'Teen'], ['adult', 'Adult'], ['unknown', 'Not sure']],
    role: [['listener', 'Listening'], ['asker', 'Asking'], ['participant', 'Taking part'], ['nearby', 'Just nearby']],
  };
  const LABELS = { relationship: 'Who they are', age_band: 'Age', role: 'Doing' };
  const FIELD_CLASS = 'w-full rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[11px]';
  const LABEL_SPAN = '<span class="block text-[10px] uppercase tracking-wide text-gray-500">';

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }

  function selectHtml(field, idx, current) {
    const opts = CHOICES[field].map(function (pair) {
      return '<option value="' + pair[0] + '"' + (pair[0] === current ? ' selected' : '') + '>' + pair[1] + '</option>';
    }).join('');
    return '<label class="space-y-0.5">' + LABEL_SPAN + LABELS[field] + '</span>'
      + '<select data-presence-field="' + field + '" data-presence-index="' + idx + '" class="' + FIELD_CLASS + '">' + opts + '</select></label>';
  }

  function rowHtml(item, idx) {
    const c = item || {};
    const relationship = c.relationship || 'other';
    return '<label class="col-span-2 space-y-0.5">' + LABEL_SPAN + 'Name</span>'
      + '<input data-presence-field="display_name" data-presence-index="' + idx + '" class="' + FIELD_CLASS + '" placeholder="e.g. Pip" value="' + escapeHtml(c.display_name) + '" /></label>'
      + selectHtml('relationship', idx, relationship)
      + selectHtml('age_band', idx, c.age_band || (relationship === 'child' ? 'child' : 'unknown'))
      + selectHtml('role', idx, c.role || 'listener')
      + '<label class="col-span-2 space-y-0.5 md:col-span-5">' + LABEL_SPAN + 'Note (optional)</span>'
      + '<input data-presence-field="context_note" data-presence-index="' + idx + '" class="' + FIELD_CLASS + '" placeholder="e.g. loves dinosaurs" value="' + escapeHtml(c.context_note) + '" /></label>'
      + '<button data-presence-remove="' + idx + '" class="rounded border border-rose-500/50 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-200" type="button">Remove</button>';
  }

  // fields: [{dataset: {presenceIndex, presenceField}, value}] in DOM order.
  function collectRows(fields) {
    const rows = {};
    const order = [];
    (fields || []).forEach(function (el) {
      const idx = String((el.dataset && el.dataset.presenceIndex) || '');
      const field = String((el.dataset && el.dataset.presenceField) || '');
      if (!field) return;
      if (!rows[idx]) { rows[idx] = {}; order.push(idx); }
      rows[idx][field] = String(el.value || '').trim();
    });
    return order.map(function (idx) { return rows[idx]; }).filter(function (row) { return row.display_name; });
  }

  const api = { CHOICES: CHOICES, escapeHtml: escapeHtml, rowHtml: rowHtml, collectRows: collectRows };

  global.OrionPresenceForm = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

/* Regression test for a live incident (2026-09-09): the modal's base CSS
 * rule (`#runtimeActivityModal { display: flex; ... }`) set `display`
 * unconditionally. A browser's own default stylesheet hides a `hidden`
 * element with a single low-specificity attribute selector
 * (`[hidden] { display: none }`), and an ID selector always outranks a bare
 * attribute selector -- so the author rule silently won regardless of the
 * `hidden` attribute, and the modal covered the whole page permanently with
 * no way to close it.
 *
 * This test does not render anything (jsdom's CSS cascade support is too
 * partial to trust for this). It parses the real stylesheet text and checks
 * the actual mechanism: every rule that sets `display` on
 * `#runtimeActivityModal` without a `[hidden]` qualifier must have LOWER
 * CSS specificity than the `[hidden]` override rule that undoes it, so the
 * override always wins the cascade regardless of source order.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const CSS_PATH = path.join(__dirname, '..', 'css', 'runtime-activity.css');
const MODAL_ID = 'runtimeActivityModal';

/* Minimal CSS specificity: (ids, classes/attrs/pseudo-classes, elements). */
function specificity(selector) {
  const ids = (selector.match(/#[\w-]+/g) || []).length;
  const classesAndAttrs = (selector.match(/\.[\w-]+|\[[^\]]*\]|:(?!:)[\w-]+/g) || []).length;
  const elements = (selector.match(/(^|[\s>+~,(])[a-zA-Z][\w-]*/g) || []).length;
  return [ids, classesAndAttrs, elements];
}

function cmp(a, b) {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

/* Every top-level `selector { body }` block in the file, in source order. */
function parseRules(css) {
  const rules = [];
  const re = /([^{}]+)\{([^{}]*)\}/g;
  let m;
  while ((m = re.exec(css))) {
    rules.push({ selectors: m[1].trim(), body: m[2] });
  }
  return rules;
}

test('a display-setting rule on #runtimeActivityModal without [hidden] can never outrank the [hidden] override', () => {
  const css = fs.readFileSync(CSS_PATH, 'utf8');
  const rules = parseRules(css);

  const overrideRule = rules.find(
    (r) => r.selectors.includes(`#${MODAL_ID}[hidden]`) && /display\s*:\s*none/.test(r.body)
  );
  assert.ok(overrideRule, `expected a "#${MODAL_ID}[hidden] { display: none }" rule in ${CSS_PATH}`);
  const overrideSpecificity = specificity(`#${MODAL_ID}[hidden]`);

  const displaySettingRules = rules.filter(
    (r) =>
      r.selectors
        .split(',')
        .some((sel) => sel.trim().replace(/\s+/g, ' ') === `#${MODAL_ID}`) && /display\s*:/.test(r.body)
  );
  assert.ok(displaySettingRules.length > 0, 'expected the base modal rule to still set display (sanity check)');

  for (const rule of displaySettingRules) {
    const s = specificity(rule.selectors);
    assert.ok(
      cmp(overrideSpecificity, s) > 0,
      `"${rule.selectors}" (specificity ${s}) must not outrank or tie the [hidden] override (${overrideSpecificity}) -- ` +
        'otherwise a hidden modal renders visible again, exactly like the 2026-09-09 incident'
    );
  }
});

test('specificity() ranks selectors the way the incident actually played out', () => {
  // Sanity check on the comparator itself, independent of the real file:
  // this is the exact ranking that let the bug happen and the fix rely on.
  assert.ok(cmp(specificity('#x'), specificity('[hidden]')) > 0, 'a bare id beats a bare attribute selector');
  assert.ok(cmp(specificity('#x[hidden]'), specificity('#x')) > 0, 'id+attribute beats the bare id');
});

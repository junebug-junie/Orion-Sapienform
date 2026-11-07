const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const oa = require("./orion-abilities.js");

const { composeMessage, composeInsert, insertAtCaret, filterCatalog, countEntries } = oa;

const CATALOG = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "catalog", "orion-abilities.json"), "utf8"),
);

test("composeMessage puts the hint on its own line after the ask", () => {
  assert.equal(composeMessage({ message: "Please look.", hint: "(dig: `x`)" }), "Please look.\n(dig: `x`)");
  assert.equal(composeMessage({ message: "  Just ask.  " }), "Just ask.");
});

test("composeInsert lands the caret after a trailing colon, before the hint", () => {
  const r = composeInsert({ message: "Read this page: ", hint: "(tool: `x`)" });
  assert.equal(r.text, "Read this page: \n(tool: `x`)");
  assert.equal(r.text.slice(0, r.caretIn), "Read this page: ");
  const plain = composeInsert({ message: "Look.", hint: "(h: `y`)" });
  assert.equal(plain.caretIn, plain.text.length);
});

test("insertAtCaret into an empty box is the text alone", () => {
  const r = insertAtCaret("", 0, "Hello Orion.");
  assert.equal(r.value, "Hello Orion.");
  assert.equal(r.caret, "Hello Orion.".length);
});

test("insertAtCaret never deletes typed text and separates it with a blank line", () => {
  const r = insertAtCaret("I was thinking", 14, "Look at your body.");
  assert.equal(r.value, "I was thinking\n\nLook at your body.");
  assert.equal(r.caret, r.value.length);
});

test("insertAtCaret in the middle keeps both halves", () => {
  const value = "before\nafter";
  const r = insertAtCaret(value, 6, "MID");
  assert.equal(r.value, "before\n\nMID\n\nafter");
  assert.ok(r.value.startsWith("before") && r.value.endsWith("after"));
  assert.equal(r.value.slice(0, r.caret), "before\n\nMID");
  assert.equal(r.value.slice(r.start, r.caret), "MID");
});

test("insertAtCaret with a bad caret appends at the end", () => {
  assert.equal(insertAtCaret("typed", 99, "x").value, "typed\n\nx");
  assert.equal(insertAtCaret("typed", null, "x").value, "typed\n\nx");
});

test("insertAtCaret drops trailing blank space in an otherwise empty box", () => {
  assert.equal(insertAtCaret("  \n", 3, "x").value, "x");
});

test("filterCatalog with no query keeps every entry", () => {
  const all = filterCatalog(CATALOG, "");
  const total = CATALOG.sections.reduce((n, s) => n + s.entries.length, 0);
  assert.equal(countEntries(all), total);
});

test("filterCatalog matches every word, across title, description and message", () => {
  const groups = filterCatalog(CATALOG, "portrait");
  assert.ok(countEntries(groups) >= 1);
  groups.forEach((g) => g.entries.forEach((e) => {
    const hay = [g.section.title, g.section.blurb, e.title, e.description, e.message, e.hint, ...(e.keywords || [])]
      .join(" ").toLowerCase();
    assert.ok(hay.includes("portrait"));
  }));
  assert.equal(countEntries(filterCatalog(CATALOG, "zzzz-no-such-thing")), 0);
});

test("every catalog entry composes to a plain-English message, not a bare command", () => {
  CATALOG.sections.forEach((s) => s.entries.forEach((e) => {
    const msg = composeMessage(e);
    assert.ok(msg.length > 10, e.id);
    assert.ok(!/^(python|`|\$|mcp__)/.test(msg), `${e.id} starts with a command`);
  }));
});

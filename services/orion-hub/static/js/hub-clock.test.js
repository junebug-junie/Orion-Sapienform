const test = require("node:test");
const assert = require("node:assert/strict");
const { clockText } = require("./hub-clock.js");

test("winter: UTC and MST, 7 hours apart", () => {
  assert.equal(clockText(new Date(Date.UTC(2026, 10, 25, 0, 7))), "12:07am UTC / 5:07pm MST");
});

test("summer: UTC and MDT, 6 hours apart", () => {
  assert.equal(clockText(new Date(Date.UTC(2026, 6, 4, 18, 30))), "6:30pm UTC / 12:30pm MDT");
});

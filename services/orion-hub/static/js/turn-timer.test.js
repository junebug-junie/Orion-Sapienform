const test = require('node:test');
const assert = require('node:assert');

const { formatTurnElapsed } = require('./turn-timer.js');

test('sub-minute durations render tenths of a second', () => {
  assert.strictEqual(formatTurnElapsed(0), '0.0s');
  assert.strictEqual(formatTurnElapsed(1500), '1.5s');
  assert.strictEqual(formatTurnElapsed(59949), '59.9s');
});

test('the minute boundary rolls over instead of reading 60.0s', () => {
  assert.strictEqual(formatTurnElapsed(60000), '1m 00s');
  assert.strictEqual(formatTurnElapsed(61000), '1m 01s');
});

test('seconds past a minute are zero-padded to two digits', () => {
  assert.strictEqual(formatTurnElapsed(65000), '1m 05s');
  assert.strictEqual(formatTurnElapsed(125000), '2m 05s');
  assert.strictEqual(formatTurnElapsed(3600000), '60m 00s');
});

test('seconds are truncated, not rounded, so the display never overshoots', () => {
  // 1m 59.9s must not print "1m 60s".
  assert.strictEqual(formatTurnElapsed(119900), '1m 59s');
});

test('a backwards clock step renders zero, not NaN or a negative duration', () => {
  assert.strictEqual(formatTurnElapsed(-5000), '0.0s');
  assert.strictEqual(formatTurnElapsed(NaN), '0.0s');
  assert.strictEqual(formatTurnElapsed(undefined), '0.0s');
});

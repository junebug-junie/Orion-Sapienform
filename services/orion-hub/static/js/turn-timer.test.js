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

test('a value that ROUNDS to 60.0s rolls over too', () => {
  // The original fixtures stopped at 59949 -- one millisecond below the break --
  // so they were written to the code rather than to the claim, and 59950 really
  // did render the forbidden "60.0s".
  assert.strictEqual(formatTurnElapsed(59949), '59.9s');
  assert.strictEqual(formatTurnElapsed(59950), '1m 00s');
  assert.strictEqual(formatTurnElapsed(59999), '1m 00s');
});

test('no input anywhere near a minute boundary can render a bare 60.0s', () => {
  for (let ms = 59000; ms <= 61000; ms += 1) {
    assert.notStrictEqual(formatTurnElapsed(ms), '60.0s');
  }
  // Same trap one minute up.
  for (let ms = 119000; ms <= 121000; ms += 1) {
    const out = formatTurnElapsed(ms);
    assert.notStrictEqual(out, '1m 60s');
  }
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

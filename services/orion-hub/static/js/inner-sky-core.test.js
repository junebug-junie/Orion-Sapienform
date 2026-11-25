const test = require('node:test');
const assert = require('node:assert/strict');
const core = require('./inner-sky-core.js');

test('insertSorted keeps order and returns the index used, ties after existing rows', () => {
  const arr = [{ t: 1 }, { t: 3 }, { t: 3 }, { t: 5 }];
  const late = { t: 3, late: true };
  assert.equal(core.insertSorted(arr, late, (e) => e.t), 3);
  assert.deepEqual(arr.map((e) => e.t), [1, 3, 3, 3, 5]);
  assert.equal(arr[3], late);
  assert.equal(core.insertSorted(arr, { t: 0 }, (e) => e.t), 0);
  assert.equal(core.insertSorted(arr, { t: 9 }, (e) => e.t), arr.length - 1);
});

test('countRates: first reading and a reset counter give no rate, never a guess', () => {
  const key = (l) => l.a + '>' + l.b;
  const first = core.countRates(null, [{ a: 'x', b: 'y', n: 100 }], key, 1000);
  assert.deepEqual(first.rates, {});
  const second = core.countRates(first.next, [{ a: 'x', b: 'y', n: 160 }, { a: 'p', b: 'q', n: 5 }], key, 1020);
  assert.equal(second.rates['x>y'], 3);
  assert.equal('p>q' in second.rates, false);
  const reset = core.countRates(second.next, [{ a: 'x', b: 'y', n: 10 }], key, 1040);
  assert.equal('x>y' in reset.rates, false);
});

test('countRates: an idle link reads zero, which is calm, not missing', () => {
  const key = (s) => s.id;
  const a = core.countRates(null, [{ id: 's', n: 7 }], key, 0);
  const b = core.countRates(a.next, [{ id: 's', n: 7 }], key, 20);
  assert.equal(b.rates.s, 0);
});

test('countDeltas: one light per real message; first reading and a reset counter draw nothing', () => {
  const key = (l) => l.a + '>' + l.b;
  const first = core.countDeltas(null, [{ a: 'x', b: 'y', n: 100 }], key);
  assert.deepEqual(first.deltas, {});
  const second = core.countDeltas(first.next, [{ a: 'x', b: 'y', n: 107 }, { a: 'new', b: 'z', n: 3 }], key);
  assert.deepEqual(second.deltas, { 'x>y': 7 });
  const reset = core.countDeltas(second.next, [{ a: 'x', b: 'y', n: 2 }], key);
  assert.deepEqual(reset.deltas, {});
});

test('binCounts drops events outside the window', () => {
  const ev = [{ t: -1 }, { t: 0 }, { t: 4.9 }, { t: 5 }, { t: 9.99 }, { t: 10 }];
  assert.deepEqual(core.binCounts(ev, 0, 10, 2), [2, 2]);
});

test('isLive uses the server clock and treats a never-seen service as dead', () => {
  assert.equal(core.isLive({ last: 990 }, 1000, 3600), true);
  assert.equal(core.isLive({ last: 990 - 3600 }, 1000, 3600), false);
  assert.equal(core.isLive({ last: null }, 1000, 3600), false);
});

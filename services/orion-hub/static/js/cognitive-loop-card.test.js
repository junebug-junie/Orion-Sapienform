const test = require('node:test');
const assert = require('node:assert/strict');
const cognitiveLoopCard = require('./cognitive-loop-card.js');

test('resolvable card shows actions, no chronic badge/note', () => {
  const vm = cognitiveLoopCard.cognitiveLoopCardViewModel({ card_kind: 'resolvable', recurrence_count: 2 });
  assert.equal(vm.isChronic, false);
  assert.equal(vm.borderClass, 'border-purple-800');
  assert.equal(vm.showChronicBadge, false);
  assert.equal(vm.showActions, true);
  assert.equal(vm.chronicNoteText, null);
});

test('chronic_pressure card hides actions, shows badge and recurrence note', () => {
  const vm = cognitiveLoopCard.cognitiveLoopCardViewModel({ card_kind: 'chronic_pressure', recurrence_count: 5000 });
  assert.equal(vm.isChronic, true);
  assert.equal(vm.borderClass, 'border-amber-800');
  assert.equal(vm.showChronicBadge, true);
  assert.equal(vm.showActions, false);
  assert.equal(vm.chronicNoteText, 'Recurring 5000x -- ongoing system state, not a pending decision.');
});

test('missing recurrence_count defaults to 0 in the chronic note, not "undefined"', () => {
  const vm = cognitiveLoopCard.cognitiveLoopCardViewModel({ card_kind: 'chronic_pressure' });
  assert.equal(vm.chronicNoteText, 'Recurring 0x -- ongoing system state, not a pending decision.');
});

test('an unrecognized card_kind is treated as resolvable, not chronic', () => {
  // Matches the backend's allowlist-only-chat philosophy from the OTHER
  // direction: on the render side, only the literal 'chronic_pressure' string
  // suppresses actions -- an unexpected value must not silently hide the
  // human's ability to act (the server-side 409 guard is what enforces
  // safety when the true scope IS ambiguous; the client isn't the safety net).
  const vm = cognitiveLoopCard.cognitiveLoopCardViewModel({ card_kind: 'something_new' });
  assert.equal(vm.isChronic, false);
  assert.equal(vm.showActions, true);
});

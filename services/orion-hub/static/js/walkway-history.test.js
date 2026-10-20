const test = require('node:test');
const assert = require('node:assert/strict');
const wh = require('./walkway-history.js');

test('camera chip says minutes for an hour and hours for longer periods', () => {
  assert.equal(wh.cameraChip({ camera: { up_minutes: 34, expected_minutes: 60, status: 'partial' } }),
    'camera partly down 34/60 min');
  assert.equal(wh.cameraChip({ camera: { up_minutes: 1212, expected_minutes: 1440, status: 'ok' } }),
    'camera up 20.2/24.0 h');
  assert.equal(wh.cameraChip({ camera: { status: 'down', up_minutes: 0, expected_minutes: 60 } }),
    'camera down 0/60 min');
  assert.equal(wh.cameraChip({}), 'camera ?');
});

test('captions are labelled as one-frame captions, blank ones dropped, at most three', () => {
  const lines = wh.captionLines({ captions: [{ at: '14:23', text: 'A dog.' }, { text: '  ' }, { at: '1', text: 'b' },
    { at: '2', text: 'c' }, { at: '3', text: 'd' }] });
  assert.equal(lines.length, 3);
  assert.equal(lines[0], 'caption 14:23 (one frame, may be wrong): A dog.');
});

test('grains are hour, day, week, month', () => {
  assert.deepEqual(wh.GRAINS.map((g) => g.value), ['hour', 'day', 'week', 'month']);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  LIVE_FRAME_VIEWS,
  formatAge,
  liveFrameState,
  shouldFetchImage,
  imageUrl,
} = require('./vision-live-frame.js');

test('walkway is a live view on the walkway stream, polled faster than carbon', () => {
  assert.equal(LIVE_FRAME_VIEWS['walkway-live'].streamId, 'walkway');
  assert.equal(LIVE_FRAME_VIEWS['carbon-live'].streamId, 'carbon');
  assert.ok(LIVE_FRAME_VIEWS['walkway-live'].pollMs <= 2000);
});

test('no frame yet shows a message naming the camera, not an image', () => {
  const s = liveFrameState({ available: false, reason: 'no_frame_seen_yet' }, LIVE_FRAME_VIEWS['walkway-live']);
  assert.equal(s.kind, 'none');
  assert.match(s.message, /No frame from Walkway yet/);
  assert.match(s.message, /orion-vision-retina-walkway/);
});

test('failed fetch (null meta) is treated as no frame', () => {
  assert.equal(liveFrameState(null, LIVE_FRAME_VIEWS['walkway-live']).kind, 'none');
});

test('a stale frame is badged STALE and dimmed, never LIVE', () => {
  const s = liveFrameState({ available: true, frame_id: 'a.jpg', age_sec: 23.4, stale: true }, LIVE_FRAME_VIEWS['walkway-live']);
  assert.equal(s.kind, 'stale');
  assert.match(s.badgeText, /^STALE/);
  assert.match(s.badgeText, /23s ago/);
  assert.match(s.imageClass, /opacity-40/);
});

test('a fresh frame is badged LIVE and not dimmed', () => {
  const s = liveFrameState({ available: true, frame_id: 'a.jpg', age_sec: 0.4, stale: false }, LIVE_FRAME_VIEWS['walkway-live']);
  assert.equal(s.kind, 'live');
  assert.match(s.badgeText, /^LIVE/);
  assert.equal(s.imageClass, '');
});

test('formatAge handles missing and large values', () => {
  assert.equal(formatAge(null), 'unknown');
  assert.equal(formatAge(0.2), 'just now');
  assert.equal(formatAge(600), '10m ago');
  assert.equal(formatAge(7200), '2h ago');
});

test('image bytes are only refetched when the frame changed', () => {
  assert.equal(shouldFetchImage({ available: true, frame_id: 'b.jpg' }, 'a.jpg'), true);
  assert.equal(shouldFetchImage({ available: true, frame_id: 'a.jpg' }, 'a.jpg'), false);
  assert.equal(shouldFetchImage({ available: false }, null), false);
  assert.equal(shouldFetchImage(null, null), false);
});

test('image URL hits the generic route with the stream id', () => {
  const url = imageUrl('walkway', 'x y.jpg');
  assert.match(url, /^\/api\/vision\/frames\/latest\/image\?/);
  assert.match(url, /stream_id=walkway/);
  assert.match(url, /f=x\+y\.jpg/);
});

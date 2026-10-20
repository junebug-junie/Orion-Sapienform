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
  assert.equal(LIVE_FRAME_VIEWS['doorbell-live'].streamId, 'doorbell');
  assert.ok(LIVE_FRAME_VIEWS['walkway-live'].pollMs <= 2000);
});

const W = LIVE_FRAME_VIEWS['walkway-live'];
const NOW = 1_000_000;
const fresh = { available: true, frame_id: 'a.jpg', age_sec: 0.4, stale: false, stale_after_sec: 10 };

test('no frame yet and nothing on screen shows a message naming the camera', () => {
  const s = liveFrameState({ available: false, reason: 'no_frame_seen_yet' }, W, {}, NOW);
  assert.equal(s.kind, 'none');
  assert.match(s.message, /No frame from Walkway yet/);
  assert.match(s.message, /orion-vision-retina-walkway/);
});

test('failed fetch with nothing on screen says the feed is unreachable', () => {
  const s = liveFrameState(null, W, {}, NOW);
  assert.equal(s.kind, 'none');
  assert.match(s.message, /Can't reach/);
});

test('a failed poll keeps the picture but badges it STALE, not LIVE', () => {
  const s = liveFrameState(null, W, { loadedAtMs: NOW - 500 }, NOW);
  assert.equal(s.kind, 'stale');
  assert.match(s.badgeText, /connection lost/);
});

test('a stale stream is badged STALE and dimmed, never LIVE', () => {
  const s = liveFrameState({ ...fresh, age_sec: 23.4, stale: true }, W, { loadedAtMs: NOW - 500 }, NOW);
  assert.equal(s.kind, 'stale');
  assert.match(s.badgeText, /^STALE/);
  assert.match(s.badgeText, /23s ago/);
  assert.match(s.imageClass, /opacity-40/);
});

test('fresh backend + recently loaded picture is LIVE', () => {
  const s = liveFrameState(fresh, W, { loadedAtMs: NOW - 1000 }, NOW);
  assert.equal(s.kind, 'live');
  assert.match(s.badgeText, /^LIVE/);
  assert.equal(s.imageClass, '');
});

test('fresh backend but the image never loaded is not LIVE (review finding)', () => {
  assert.equal(liveFrameState(fresh, W, {}, NOW).kind, 'loading');
  const err = liveFrameState(fresh, W, { lastError: true }, NOW);
  assert.equal(err.kind, 'stale');
  assert.match(err.badgeText, /IMAGE ERROR/);
});

test('fresh backend but the picture stopped updating is STALE (review finding)', () => {
  const s = liveFrameState(fresh, W, { loadedAtMs: NOW - 30_000, lastError: true }, NOW);
  assert.equal(s.kind, 'stale');
  assert.match(s.badgeText, /picture 30s ago/);
  assert.match(s.badgeText, /image error/);
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

const test = require('node:test');
const assert = require('node:assert/strict');
const selfBrain = require('./self-brain.js');

test('hitTestRegion picks the region whose circle contains the point', () => {
  const hitboxes = [
    { cx: 10, cy: 10, radius: 5, region: { region_id: 'a' } },
    { cx: 100, cy: 100, radius: 5, region: { region_id: 'b' } },
  ];
  const hit = selfBrain.hitTestRegion(hitboxes, 11, 9);
  assert.equal(hit.region.region_id, 'a');
});

test('hitTestRegion returns null when the point is outside every hitbox', () => {
  const hitboxes = [{ cx: 10, cy: 10, radius: 5, region: { region_id: 'a' } }];
  assert.equal(selfBrain.hitTestRegion(hitboxes, 500, 500), null);
});

test('hitTestRegion picks the nearest hitbox when two overlap', () => {
  const hitboxes = [
    { cx: 10, cy: 10, radius: 20, region: { region_id: 'far-but-big' } },
    { cx: 12, cy: 10, radius: 20, region: { region_id: 'near' } },
  ];
  const hit = selfBrain.hitTestRegion(hitboxes, 12, 10);
  assert.equal(hit.region.region_id, 'near');
});

test('hitTestRegion handles an empty hitbox list', () => {
  assert.equal(selfBrain.hitTestRegion([], 0, 0), null);
});

test('fmtDetailValue formats a non-integer float to 4 decimal places', () => {
  assert.equal(selfBrain.fmtDetailValue(0.0047855289), '0.0048');
});

test('fmtDetailValue leaves an integer-valued number bare', () => {
  assert.equal(selfBrain.fmtDetailValue(30), '30');
});

test('fmtDetailValue stringifies a non-number as-is', () => {
  assert.equal(selfBrain.fmtDetailValue('steady'), 'steady');
});

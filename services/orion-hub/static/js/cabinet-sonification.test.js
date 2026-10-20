const {test} = require('node:test');
const assert = require('node:assert/strict');
const {PRESET, frameFromPayload, mapFrame, Onset, Instrument} = require('./cabinet-sonification');
const now = Date.parse('2026-10-20T18:00:00Z');
function payload(at = now, value = 40) {
  return {ok: true, age_sec: 0, snapshot: {status: 'ok'}, sources: {a: {snapshot: {status: 'ok', received_at: new Date(at).toISOString(), frame: {
    magnetic: {magnitude_ut: value}, lidar: {distance_mm: 0, status: 0}, imu: {accel_x: 0, accel_y: 0, accel_z: 9.80665}
  }}}}};
}
const approved = {...PRESET, voices: PRESET.voices.map(v => ({...v, approved: true}))};
test('deterministic native units, zero, log pitch, near-fast pulses, clipping and gates', () => {
  const f = frameFromPayload(payload(), now);
  assert.deepEqual(mapFrame(f), mapFrame(f));
  assert.equal(f.readings[2].value, 0);
  assert.equal(mapFrame(f)[0].frequencyHz, 220);
  assert.equal(mapFrame(f)[1].enabled, false);
  assert.equal(mapFrame(f, approved)[1].pulseHz, 4);
  const clipped = mapFrame(frameFromPayload(payload(now, 80), now))[0];
  assert.equal(clipped.frequencyHz, 440); assert.equal(clipped.clipped, true);
  for (const value of [NaN, Infinity, null, '', true, -1]) assert.equal(mapFrame(frameFromPayload(payload(now, value), now))[0].enabled, false);
  f.readings[0].unit = 'wrong'; assert.equal(mapFrame(f)[0].reason, 'invalid unit');
  assert.equal(mapFrame(frameFromPayload(payload(), now), {...PRESET, voices: [{...PRESET.voices[0], hi: 20}]})[0].reason, 'invalid preset');
});
test('newest valid channel wins; frozen dual source cannot borrow fresh aggregate age', () => {
  const p = payload(now - 6000);
  p.sources.b = payload().sources.a;
  p.sources.b.snapshot.frame = {lidar: {distance_mm: 150, status: 0}};
  let f = frameFromPayload(p, now);
  assert.equal(f.readings[0].status, 'stale'); assert.equal(f.readings[1].sourceId, 'b');
  p.sources.a = payload(now - 1000).sources.a;
  p.sources.b.snapshot.frame.magnetic = {magnitude_ut: 51};
  f = frameFromPayload(p, now); assert.equal(f.readings[0].value, 51);
  p.sources.b.snapshot.status = 'error';
  assert.equal(frameFromPayload(p, now).readings[0].sourceId, 'a');
  p.sources.b.snapshot.status = 'ok'; p.sources.b.snapshot.received_at = p.sources.a.snapshot.received_at;
  assert.equal(frameFromPayload(p, now).readings[0].sourceId, 'a');
  p.sources.a.snapshot.frame.lidar.status = 1; p.sources.b.snapshot.frame.lidar.status = 1;
  assert.equal(frameFromPayload(p, now).readings[1].status, 'invalid');
  p.sources = {}; assert.equal(frameFromPayload(p, now).readings[0].status, 'missing');
});
test('server freshness, future, malformed timestamps and non-finite sensor axes fail closed', () => {
  for (const change of [p => p.ok = false, p => p.age_sec = 9, p => p.age_sec = null,
    p => p.sources.a.snapshot.received_at = 'bad', p => p.sources.a.snapshot.received_at = new Date(now + 1000).toISOString()]) {
    const p = payload(); change(p); assert.equal(mapFrame(frameFromPayload(p, now))[0].enabled, false);
  }
  const p = payload(); p.sources.a.snapshot.frame.imu.accel_z = null;
  assert.equal(frameFromPayload(p, now).readings[2].status, 'invalid');
});
test('onsets require baseline, hysteresis, unique source time; recovery and source switch suppress hits', () => {
  const onset = new Onset(), def = PRESET.voices[2];
  const r = (t, value, sourceId = 'b') => ({observedAt: new Date(now + t).toISOString(), status: 'ok', sourceId, value});
  assert.equal(onset.update(r(0, .2), def), false);
  assert.equal(onset.update(r(1000, .04), def), false);
  assert.equal(onset.update(r(2000, .2), def), true);
  assert.equal(onset.update(r(2000, .2), def), false);
  assert.equal(onset.update(r(3000, .15), def), false);
  assert.equal(onset.update(r(4000, .04), def), false);
  assert.equal(onset.update(r(5000, .2, 'a'), def), false);
  onset.update(null, def); assert.equal(onset.update(r(6000, .2), def), false);
  onset.update(r(7000, .04), def); assert.equal(onset.update(r(13000, .2), def), false);
});
function fakeContext() {
  const param = () => ({value: 0, cancelScheduledValues() {}, setValueAtTime(v) {this.value = v;}, linearRampToValueAtTime(v) {this.value = v;}});
  return {state: 'running', currentTime: 0, destination: {}, async resume() {}, async close() {this.state = 'closed';},
    createGain() {return {gain: param(), connect() {}, disconnect() {}};},
    createOscillator() {return {frequency: param(), connect() {}, disconnect() {}, start() {}, stop() {}};}};
}
test('listen, mute, solo, trace cap, watchdog, stop and suspended recovery', async () => {
  let time = now;
  const context = fakeContext(), instrument = new Instrument({contextFactory: () => context, now: () => time});
  instrument.receive(payload()); assert.equal(instrument.context, null);
  await instrument.listen(); assert.equal(instrument.armed, true); assert.equal(instrument.master.gain.value, .35);
  instrument.mute('magnetic', true); instrument.tick(); assert.equal(instrument.armed, true); assert.equal(instrument.toneGain.gain.value, 0);
  instrument.mute('magnetic', false); instrument.setSolo('lidar'); instrument.tick(); assert.equal(instrument.armed, true); assert.equal(instrument.toneGain.gain.value, 0);
  instrument.setSolo('lidar'); instrument.startTrace();
  for (let i = 0; i < 605; i++) instrument.update();
  assert.equal(instrument.takeTrace().records.length, 600); assert.equal(instrument.trace, null);
  time += 5000; const master = instrument.master; instrument.tick();
  assert.equal(instrument.armed, false); assert.equal(master.gain.value, 0); assert.equal(instrument.timer, null);
  const suspended = new Instrument({contextFactory: () => ({...fakeContext(), state: 'suspended'})});
  await suspended.listen(); assert.equal(suspended.armed, false);
});
test('stop during asynchronous resume cannot re-arm', async () => {
  const context = fakeContext(); let finish;
  context.resume = () => new Promise(resolve => finish = resolve);
  const instrument = new Instrument({contextFactory: () => context});
  const start = instrument.listen(); instrument.stop(); finish(); await start;
  assert.equal(instrument.armed, false); assert.equal(instrument.timer, null);
});
module.exports = {payload, approved};

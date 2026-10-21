const {test} = require('node:test');
const assert = require('node:assert/strict');
const History = require('./cabinet-sonification-history');
const {Instrument, PRESET, mapFrame} = require('./cabinet-sonification');
const end = Date.parse('2026-10-20T18:00:00Z');
const point = (seconds, v) => ({t: new Date(end + seconds * 1000).toISOString(), v});
function history(window = '24h') {
  return {ok: true, window, grain_sec: 30, series: {magnetic_ut: [point(-60, 20), point(0, 60)],
    lidar_mm: [point(-45, 0), point(-15, 100)], vibration_g: [point(-60, .04), point(0, 1)]}, stats: {magnetic_ut: {n_raw: 3}}};
}
function context() {
  const param = () => ({value: 0, ramps: [], cancelScheduledValues() {}, setValueAtTime(v) {this.value = v;}, linearRampToValueAtTime(v, t) {this.value = v; this.ramps.push({v, t});}});
  return {state: 'running', currentTime: 0, destination: {}, async resume() {}, async close() {this.state = 'closed';},
    createGain() {return {gain: param(), connect() {}, disconnect() {}};},
    createOscillator() {return {frequency: param(), connect() {}, disconnect() {}, start() {}, stop() {}};}};
}
test('24h/3d/7d compression, sorted snapshots and fixed bounds', () => {
  for (const [window, compression] of [['24h',1440],['3d',4320],['7d',10080]]) {
    const input = history(window), h = History.prepare(input, end);
    assert.equal(h.compression, compression); assert.equal(h.end - h.start, compression * 60000);
    const f = History.frame(h, 60 - 30 / compression, 90, PRESET);
    assert.equal(f.readings[0].value, 40);
    assert.equal(f.readings[1].value, 50); // independent bracketing, not magnetic's indexes
    assert.equal(f.readings[2].value, null);
    assert.match(f.readings[2].reason, /omitted/);
    assert.equal(mapFrame(f)[0].frequencyHz, 220);
    assert.equal(f.readings[0].observedAt, point(-60, 0).t);
    assert.equal(f.sourceTime, point(-30, 0).t);
    input.series.magnetic_ut[0].v = 999; assert.equal(h.series.magnetic_ut[0].v, 20);
    assert.equal(h.stats.magnetic_ut.downsampled, true);
  }
});
test('gaps are silence, explicit wider policy, no extrapolation and invalid barriers', () => {
  const input = history(); input.series.magnetic_ut = [point(-180, 0), point(0, 60)];
  let h = History.prepare(input, end), at = 60 - 90 / 1440;
  assert.equal(History.frame(h, at, 90, PRESET).readings[0].value, null);
  assert.equal(History.frame(h, at, 300, PRESET).readings[0].value, 30);
  assert.equal(h.stats.magnetic_ut.maxSpacingSec, 180);
  assert.equal(History.frame(h, 0, 300, PRESET).readings[0].value, null);
  assert.equal(History.frame(h, 61, 300, PRESET).sourceTime, null);
  assert.equal(History.interpolate(h.series.magnetic_ut, end, NaN), null);
  input.series.magnetic_ut.splice(1,0,point(-90,null)); h = History.prepare(input, end);
  assert.equal(History.frame(h, 60 - 45 / 1440, 300, PRESET).readings[0].value, null);
});
test('empty, partial, malformed, non-finite, duplicate and unsorted history', () => {
  assert.equal(History.prepare({ok:false}, end),null);
  assert.equal(History.prepare({...history(),window:'bad'}, end),null);
  assert.equal(History.prepare({...history(),series:{}}, end),null);
  const p = history(); p.series = {magnetic_ut: [point(0,60),point(-60,20),point(-30,NaN),{t:'bad',v:1},point(0,99)]};
  const h = History.prepare(p, end);
  assert.equal(h.series.magnetic_ut[0].v,20);
  assert.equal(h.series.magnetic_ut.at(-1).v,null);
  assert.equal(History.frame(h,60,90,PRESET).readings[1].status,'missing');
  assert.equal(h.stats.magnetic_ut.invalid,2);
});
test('audio-clock pause/resume/seek/replay, live poll isolation, gaps and natural end', async () => {
  let ctx;
  const instrument = new Instrument({now: () => end, contextFactory: () => (ctx = context())});
  instrument.receiveHistory(history()); instrument.setMode('history'); await instrument.listen();
  assert.equal(instrument.armed,true); // begins in a gap; clock must continue
  ctx.currentTime=12; instrument.tick(); assert.equal(instrument.frame.playbackSec,12);
  instrument.receive({ok:false}); instrument.fail(); assert.equal(instrument.armed,true);
  instrument.pause(); assert.equal(instrument.positionSec,12); assert.equal(instrument.armed,false);
  await instrument.listen(); ctx.currentTime=3; instrument.tick(); assert.equal(instrument.playbackSeconds(),15);
  instrument.seek(60-30/1440); assert.equal(instrument.armed,false); assert.equal(instrument.frame.readings[0].value,40);
  await instrument.listen(); ctx.currentTime=1; instrument.tick();
  assert.equal(instrument.armed,false); assert.equal(instrument.positionSec,60); assert.match(instrument.reason,/complete/);
  await instrument.replay(); assert.equal(instrument.playbackSeconds(),0); assert.equal(instrument.armed,true);
  instrument.setMode('live'); assert.equal(instrument.armed,false); assert.equal(instrument.mode,'live');
});
test('window changes/failures cannot keep old history playing or autoplay', async () => {
  const instrument = new Instrument({now: () => end, contextFactory: context});
  instrument.receiveHistory(history()); instrument.setMode('history'); await instrument.listen();
  instrument.historyLoading('7d'); assert.equal(instrument.armed,false); assert.equal(instrument.history,null);
  await instrument.listen(); assert.equal(instrument.armed,false);
  instrument.receiveHistory(history('7d')); assert.equal(instrument.armed,false);
  await instrument.listen(); instrument.historyFailed(); assert.equal(instrument.history,null); assert.equal(instrument.armed,false);
  instrument.receiveHistory(history()); await instrument.listen(); instrument.setGap(300);
  assert.equal(instrument.armed,false); assert.equal(instrument.gapSec,300);
  instrument.setGap(Infinity); assert.equal(instrument.gapSec,300);
});
test('gap mute ramps are not perpetually extended; Stop resets cursor and trace records policy', async () => {
  const instrument = new Instrument({now: () => end, contextFactory: context});
  instrument.receiveHistory(history()); instrument.setMode('history'); instrument.seek(60-30/1440);
  await instrument.listen(); const gain=instrument.toneGain.gain;
  instrument.context.currentTime=.01; instrument.update();
  instrument.history = History.prepare({...history(),series:{lidar_mm:[point(0,3)]}}, end);
  instrument.update(); const count=gain.ramps.length;
  for (let i=0;i<20;i++) instrument.update();
  assert.equal(gain.ramps.length,count); assert.equal(gain.ramps.at(-1).v,0);
  instrument.startTrace(); const trace=instrument.takeTrace();
  assert.equal(trace.records[0].frame.mode,'history'); assert.equal(trace.history.gapSec,90);
  instrument.stop(); assert.equal(instrument.positionSec,0); assert.equal(instrument.nodes.size,0);
});

test('request bounds retain trailing outages and cannot shift with partial channels', () => {
  const p = history(); p.series = {magnetic_ut: [point(-7*3600,20),point(-6*3600,40)]};
  const h = History.prepare(p, end);
  assert.equal(h.start,end-86400000); assert.equal(h.end,end);
  assert.equal(History.frame(h,59,3600,PRESET).readings[0].value,null);
  assert.equal(History.frame(h,60-6.5*3600/1440,3600,PRESET).readings[0].value,30);
  p.series.lidar_mm=[point(0,100)];
  const partial = History.prepare(p,end);
  assert.equal(partial.start,h.start); assert.equal(partial.end,h.end);
  assert.equal(History.frame(partial,59,3600,PRESET).readings[0].value,null);
  p.series.magnetic_ut=[point(-90000,3),point(100,4)];
  assert.equal(History.prepare(p,end).series.magnetic_ut.length,0);
  assert.equal(History.prepare(history(),NaN),null);
});

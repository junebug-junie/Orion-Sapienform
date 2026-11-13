/* Real rendered Hub controls + native Web Audio; isolated cabinet_sonification_server.py.
 * PUPPETEER_MODULE=/path/to/puppeteer node this-file http://127.0.0.1:18097
 * No live writes, microphone access or audio uploads. */
const assert = require('node:assert/strict');
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer');
const recordings = require('./fixtures/cabinet-sonification-live.json');
const historical = require('./fixtures/cabinet-sonification-history-live.json');
(async () => {
  const browser = await puppeteer.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    await page.setRequestInterception(true);
    let polls = 0, historyPolls = 0, historyFailure = false;
    page.on('request', req => {
      const url = new URL(req.url());
      if (url.pathname === '/api/cabinet/sensors/history') {
        historyPolls++;
        return req.respond({status: historyFailure ? 503 : 200, contentType: 'application/json', body: JSON.stringify({...historical, window: url.searchParams.get('window')})});
      }
      if (url.pathname === '/api/cabinet/sensors/latest') {
        polls++;
        const p = JSON.parse(JSON.stringify(recordings[polls % recordings.length]));
        p.age_sec = 0;
        for (const s of Object.values(p.sources)) s.snapshot.received_at = new Date(Date.now() - 100).toISOString();
        return req.respond({status: 200, contentType: 'application/json', body: JSON.stringify(p)});
      }
      if (url.pathname.startsWith('/api/') || url.origin !== new URL(process.argv[2]).origin) return req.respond({status: 200, contentType: 'application/json', body: '{}'});
      req.continue();
    });
    await page.goto(process.argv[2], {waitUntil: 'domcontentloaded'});
    // Expose the real Cabinet panel, isolated from unrelated modal initialization.
    await page.evaluate(() => {
      const cabinet = document.getElementById('cabinet');
      document.body.replaceChildren(cabinet); cabinet.classList.remove('hidden');
      cabinet.style.display = 'block';
      window.OrionCabinetSensors.activate();
    });
    await page.waitForFunction(() => window.OrionCabinetSonification?.frame?.readings[0]?.status === 'ok');
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.armed), false);
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => window.OrionCabinetSonification.armed);
    await page.click('[data-sound-mute="magnetic"]');
    await new Promise(r => setTimeout(r, 100));
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.armed), true);
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.targets[0].enabled), false);
    await page.click('[data-sound-mute="magnetic"]');
    await page.click('[data-sound-solo="magnetic"]');
    assert.equal(await page.$eval('[data-sound-solo="magnetic"]', b => b.getAttribute('aria-pressed')), 'true');
    await page.click('#cabinetSoundRecord');
    assert.ok(await page.evaluate(() => window.OrionCabinetSonification.trace.length > 0));
    // Dedicated unrelated audio context must survive Stop.
    await page.evaluate(async () => { window.otherAudio = new AudioContext(); await otherAudio.resume(); });
    await page.click('#cabinetSoundStop');
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.armed), false);
    assert.equal(await page.evaluate(() => otherAudio.state), 'running');
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => window.OrionCabinetSonification.armed);
    await page.evaluate(() => document.getElementById('cabinet').style.display = 'none');
    await page.waitForFunction(() => !window.OrionCabinetSonification.armed);
    await page.evaluate(() => { document.getElementById('cabinet').style.display = 'block'; });
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.armed), false);
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => window.OrionCabinetSonification.armed);
    await page.evaluate(() => window.OrionCabinetSonification.context.suspend());
    await page.waitForFunction(() => !window.OrionCabinetSonification.armed);
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => window.OrionCabinetSonification.armed);
    await page.evaluate(() => window.dispatchEvent(new Event('pagehide')));
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.armed), false);
    const replay = await page.evaluate(async rows => {
      const {frameFromPayload, mapFrame, Instrument, PRESET} = window.OrionCabinetSound;
      for (const row of rows) {
        const now = Math.max(...Object.values(row.sources).map(s => Date.parse(s.snapshot.received_at))) + 1000;
        const frame = frameFromPayload(row, now);
        if (JSON.stringify(mapFrame(frame)) !== JSON.stringify(mapFrame(frame))) throw Error('Nondeterminism');
        if (!mapFrame(frame)[0].enabled) throw Error('Recorded magnetic source unavailable');
      }
      // Test-only approval: render all voices, including unapproved calibration paths.
      const context = new OfflineAudioContext(1, 48000, 48000);
      const synth = new Instrument({preset: {...PRESET, voices: PRESET.voices.map(v => ({...v, approved: true}))}});
      synth.context = context; synth.master = context.createGain(); synth.master.gain.value = .35; synth.master.connect(context.destination);
      const tone = context.createOscillator(), gain = context.createGain();
      tone.frequency.value = 440; gain.gain.value = .12; tone.connect(gain); gain.connect(synth.master); tone.start();
      for (let i = 0; i < 4; i++) { synth.pulse(i / 4, 330, .1, .08, 'lidar'); synth.pulse(i / 4, 90, .1, .12, 'vibration'); }
      const samples = (await context.startRendering()).getChannelData(0);
      return {finite: samples.every(Number.isFinite), peak: Math.max(...samples.map(Math.abs)), frames: rows.length};
    }, recordings);
    assert.ok(replay.finite); assert.ok(replay.peak > 0 && replay.peak < 1);
    // A failed poll immediately disarms; no stale render may keep the audio alive.
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => window.OrionCabinetSonification.armed);
    await page.evaluate(() => window.OrionCabinetSonification.fail());
    assert.equal(await page.evaluate(() => window.OrionCabinetSonification.armed), false);
    await page.waitForFunction(() => window.OrionCabinetSonification.history);
    await page.select('#cabinetSoundMode', 'history');
    assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    assert.match(await page.$eval('#cabinetSoundHistoryInfo', e => e.textContent), /Gap policy: 2× median spacing \(188 s\)/);
    const seek = seconds => page.$eval('#cabinetSoundSeek', (input, value) => { input.value=value; input.dispatchEvent(new Event('input', {bubbles:true})); }, seconds);
    await seek(30);
    assert.equal(await page.evaluate(() => OrionCabinetSonification.frame.playbackSec), 30);
    // Real 24h data (~94 s spacing) must be audible under the default policy.
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => OrionCabinetSonification.armed && OrionCabinetSonification.toneGain.gain.value > 0.1);
    await page.select('#cabinetSoundGap', '4');
    assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => OrionCabinetSonification.armed && OrionCabinetSonification.toneGain.gain.value > 0.1);
    await page.click('#cabinetSoundPause');
    const paused = await page.evaluate(() => OrionCabinetSonification.positionSec);
    await new Promise(r => setTimeout(r, 100));
    assert.equal(await page.evaluate(() => OrionCabinetSonification.playbackSeconds()), paused);
    await seek(15);
    assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    await page.click('#cabinetSoundReplay');
    await page.waitForFunction(() => OrionCabinetSonification.armed);
    assert.ok(await page.evaluate(() => OrionCabinetSonification.playbackSeconds() < 1));
    await page.click('#cabinetSoundStop');
    assert.equal(await page.evaluate(() => OrionCabinetSonification.positionSec), 0);
    for (const window of ['3d','7d']) {
      const before = historyPolls;
      await page.select('#cabinetSoundWindow', window);
      await page.waitForFunction(w => OrionCabinetSonification.history?.window === w, {}, window);
      assert.equal(historyPolls, before + 1, 'history uses the existing fetch once per window change');
      assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    }
    await page.click('#cabinetSoundListen');
    await page.waitForFunction(() => OrionCabinetSonification.armed);
    await page.evaluate(() => document.getElementById('cabinet').style.display='none');
    await page.waitForFunction(() => !OrionCabinetSonification.armed);
    await page.evaluate(() => document.getElementById('cabinet').style.display='block');
    assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    historyFailure = true;
    await page.select('#cabinetSoundWindow', '24h');
    await page.waitForFunction(() => OrionCabinetSonification.historyNote === 'History unavailable');
    await page.click('#cabinetSoundListen');
    assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    await page.select('#cabinetSoundMode', 'live');
    assert.equal(await page.evaluate(() => OrionCabinetSonification.armed), false);
    console.log(JSON.stringify({passed: true, polls, historyPolls, replay, checks: ['default policy audible on real history', 'explicit gap policy', 'pause freezes clock', 'seek pauses', 'replay', 'Stop resets history', '3d/7d fetch reuse', 'history hide/reopen', 'failed window clears history', 'mode changes disarm', 'real controls and asset', 'explicit gesture', 'mute/solo', 'bounded trace', 'Stop preserves other audio', 'hide/reopen', 'suspended recovery', 'pagehide', 'recorded replay', 'offline finite unclipped mix', 'poll failure']}));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });

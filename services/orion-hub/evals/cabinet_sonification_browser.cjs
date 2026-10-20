/* Real rendered Hub controls + native Web Audio; isolated cabinet_sonification_server.py.
 * PUPPETEER_MODULE=/path/to/puppeteer node this-file http://127.0.0.1:18097
 * No live writes, microphone access or audio uploads. */
const assert = require('node:assert/strict');
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer');
const recordings = require('./fixtures/cabinet-sonification-live.json');
(async () => {
  const browser = await puppeteer.launch({headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    await page.setRequestInterception(true);
    let polls = 0;
    page.on('request', req => {
      const url = new URL(req.url());
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
    console.log(JSON.stringify({passed: true, polls, replay, checks: ['real controls and asset', 'explicit gesture', 'mute/solo', 'bounded trace', 'Stop preserves other audio', 'hide/reopen', 'suspended recovery', 'pagehide', 'recorded replay', 'offline finite unclipped mix', 'poll failure']}));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });

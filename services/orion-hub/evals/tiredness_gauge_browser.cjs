/* Orion's tiredness gauge in the Biometrics preview card: real Hub template and
 * biometrics-view.js; every API is answered from fixtures, so nothing live is read.
 * Serve with: python services/orion-hub/evals/dream_server.py --port 18091 [--env-file ...]
 * Run:        node services/orion-hub/evals/tiredness_gauge_browser.cjs http://127.0.0.1:18091
 * PUPPETEER_MODULE may point to an existing installation. */
const assert = require('node:assert/strict');
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer');
const base = process.argv[2] || 'http://127.0.0.1:18091';

// Live shape, 2026-10-09: tired past the line, but slept recently.
let pressure = {enabled: true, ready: false, too_soon: true, is_idle: false, should_sleep: false, offer_enabled: true,
  pressure: {pressure: 11.06, threshold: 3, idle_minutes: 2, idle_required_minutes: 45,
    computed_at: new Date().toISOString(), since: new Date().toISOString(),
    counts: {metacog: 35, compaction_request: 7, resonance: 1, crystallization: 1},
    new_counts: {metacog: 14, crystallization: 1}}};

(async () => {
  const browser = await puppeteer.launch({headless: true, args: ['--no-sandbox']});
  const requests = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({width: 1440, height: 1000});
    await page.setRequestInterception(true);
    page.on('request', req => {
      const url = new URL(req.url());
      if (url.pathname === '/api/dream/pressure') {
        requests.push(url.pathname);
        return req.respond({status: 200, contentType: 'application/json', body: JSON.stringify(pressure)});
      }
      if (url.pathname.startsWith('/api/biometrics/')) {
        return req.respond({status: 200, contentType: 'application/json', body: JSON.stringify({ok: false, metrics: {}})});
      }
      req.continue();
    });
    await page.goto(base + '/', {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('#ekgViewToggle');
    await page.click('#ekgViewToggle');
    await page.waitForFunction(() => document.querySelector('#orionTiredness [role="meter"]'));

    const text = await page.$eval('#orionTiredness', e => e.textContent);
    assert.match(text, /Orion's tiredness/);
    assert.match(text, /Ready to sleep · 11\.1 \/ 3\.0/);
    assert.match(text, /off the scale/);
    assert.match(text, /waiting out the 6 h minimum/);
    assert.match(text, /14 self-noticed problems · 1 new memory/);
    const meter = await page.$eval('#orionTiredness [role="meter"]', e => ({
      now: e.getAttribute('aria-valuenow'), max: e.getAttribute('aria-valuemax'),
      fill: parseFloat(e.firstElementChild.style.width), line: parseFloat(e.children[1].style.left)}));
    assert.deepEqual(meter, {now: '6', max: '6', fill: 100, line: 50});

    const clip = () => page.$eval('#orionTiredness', e => { const r = e.getBoundingClientRect();
      return {x: r.x - 8, y: r.y - 8, width: r.width + 16, height: r.height + 16}; });
    if (process.env.TIREDNESS_EVAL_SCREENSHOT) await page.screenshot({path: process.env.TIREDNESS_EVAL_SCREENSHOT, clip: await clip()});
    // Tooltip: hidden until hover, explains tiredness, reachable by keyboard focus.
    const tipVisible = () => page.$eval('#orionTirednessTip', e => getComputedStyle(e).display !== 'none');
    assert.equal(await tipVisible(), false);
    await page.hover('#orionTiredness button');
    assert.equal(await tipVisible(), true);
    assert.match(await page.$eval('#orionTirednessTip', e => e.textContent), /Repeats don't count/);
    if (process.env.TIREDNESS_EVAL_SCREENSHOT) {
      await page.screenshot({path: process.env.TIREDNESS_EVAL_SCREENSHOT.replace(/\.png$/, '-tooltip.png'), clip: await page.$eval('#biometricsPreviewContainer', e => {
        const r = e.getBoundingClientRect(); return {x: r.x, y: r.y, width: r.width, height: 260}; })});
    }
    await page.mouse.move(0, 0);
    assert.equal(await tipVisible(), false);
    await page.focus('#orionTiredness button');
    assert.equal(await tipVisible(), true);

    // The card opens the Biometrics modal on click; the help button must not.
    await page.click('#orionTiredness button');
    const modalHidden = await page.$eval('#biometricsModalRoot', e => e.classList.contains('hidden'));
    assert.equal(modalHidden, true, 'help button opened the Biometrics modal');

    // A rested reading, and an unavailable one that must not look rested.
    pressure = {...pressure, too_soon: false, pressure: {...pressure.pressure, pressure: 0, new_counts: {}}};
    await page.evaluate(() => window.OrionBiometricsView && window.OrionBiometricsView.tirednessModel);
    const rested = await page.evaluate(p => window.OrionBiometricsView.tirednessModel(p), pressure);
    assert.equal(rested.level, 'Rested');
    const down = await page.evaluate(() => window.OrionBiometricsView.tirednessModel({detail: 'dream_pressure_unavailable'}));
    assert.equal(down.available, false);

    assert.ok(requests.length >= 1);
    console.log(JSON.stringify({passed: true, checks: ['renders in preview card', 'reading and level', 'off-scale overflow',
      'waiting reason', 'plain-word new material', 'meter aria + geometry', 'tooltip hover', 'tooltip keyboard focus',
      'help click does not open modal', 'rested', 'unavailable is not rested'], pressureRequests: requests.length}));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });

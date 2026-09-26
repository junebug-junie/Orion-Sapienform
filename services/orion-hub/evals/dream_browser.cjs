/* Real Hub template + tab controller; fixture APIs isolate behavior from live writes.
 * PUPPETEER_MODULE may point to an existing installation. */
const assert = require('node:assert/strict');
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer');
const base = process.argv[2] || 'http://127.0.0.1:18091';
const now = '2026-09-26T01:00:00Z';
const pressure = {enabled:true, ready:false, too_soon:true, offer_enabled:true,
  pressure:{pressure:9,threshold:3,idle_minutes:50,idle_required_minutes:45,computed_at:now,since:now,counts:{metacog:9}}};
const arm = {offered:1,adopted:0,tested:0,adoption_rate:0,support_rate:null};
const score = {arms:{dream:arm,control:arm},verdict:'too early: need 20 / 5',unmatched_priors:0};
const cycle = id => ({cycle_id:id,started_at:now,ended_at:now,status:'completed',trigger:'pressure',replay_count:1,hypothesis_count:1,no_link_count:2,unparseable_count:1,llm_failures:0,replay:[{source_kind:'metacog',weight:1,text:'A real replay fragment',reason:'critical recall',ref_id:'metacog:1'}]});
(async () => {
  const browser = await puppeteer.launch({headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage();
    await page.setViewport({width:1440,height:1100});
    let mode = 'normal';
    const requests = [];

    await page.setRequestInterception(true);
    page.on('request', req => {
      const url = new URL(req.url());
      if (url.pathname.startsWith('/api/dream/')) {
        requests.push({path:url.pathname,search:url.search,method:req.method()});
        let data;
        if (mode === 'unavailable') return req.respond({status:503,contentType:'application/json',body:'{"detail":"unavailable"}'});
        if (url.pathname.endsWith('/pressure')) data = pressure;
        else if (url.pathname.endsWith('/scorecard')) data = score;
        else if (url.pathname.endsWith('/cycles')) data = mode === 'empty' ? {cycles:[],has_more:false} : {cycles:[cycle(url.search ? 'dc-old' : 'dc-new')],has_more:!url.search,next_cursor:{before:now,before_id:'dc-new'}};
        else data = {cycle:cycle(url.pathname.split('/').pop()),hypotheses:[{hypothesis_id:'dh-1',arm:'control',claim:'<img src=x onerror="window.injected=1"> A testable link',why:'Specific shared mechanism',ref_a:'metacog:1',ref_b:'crystallization:2',offered_at:null,expired:false}]};
        return req.respond({status:200,contentType:'application/json',body:JSON.stringify(data)});
      }
      if (url.hostname === 'cdn.tailwindcss.com') return req.continue();
      if (url.origin !== base || url.pathname.startsWith('/api/') || url.pathname === '/curiosity') {
        return req.respond({status:200,contentType:'application/json',body:'{}'});
      }
      req.continue();
    });
    await page.goto(base+'/#dream',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => !document.getElementById('dream').classList.contains('hidden') && document.getElementById('dreamDetail').textContent.includes('dh-1'));
    assert.match(await page.$eval('#dreamPressure', e => e.textContent), /waiting for sleep gates/);
    assert.match(await page.$eval('#dreamDetail', e => e.textContent), /Random control.*waiting for waking offer/s);
    assert.match(await page.$eval('#dreamDetail', e => e.textContent), /<img src=x/);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    assert.equal(await page.$('#dreamDetail img'), null);
    await page.click('#dreamOlder');
    await page.waitForFunction(() => document.getElementById('dreamDetail').textContent.includes('dc-old'));
    await page.click('#dreamNewer');
    await page.waitForFunction(() => document.getElementById('dreamDetail').textContent.includes('dc-new'));
    await page.click('#hubTabLauncherButton');
    await page.click('#hubTabButton');
    await page.waitForFunction(() => document.getElementById('dream').classList.contains('hidden'));
    const count = requests.length;
    await page.evaluate(() => window.OrionDream.refresh());
    assert.equal(requests.length, count, 'hidden tab refreshed');
    await page.click('#hubTabLauncherButton');
    await page.click('#dreamTabButton');
    await page.waitForFunction(() => document.getElementById('dreamDetail').textContent.includes('dh-1'));
    const beforeRefresh = requests.filter(r => r.path.endsWith('/pressure')).length;
    await page.click('#dreamRefresh');
    await page.waitForFunction(() => document.getElementById('dreamDetail').textContent.includes('dh-1'));
    assert.equal(requests.filter(r => r.path.endsWith('/pressure')).length, beforeRefresh+1, 'duplicate refresh listener');
    if (process.env.DREAM_EVAL_SCREENSHOT) await page.screenshot({path:process.env.DREAM_EVAL_SCREENSHOT,fullPage:false});
    mode = 'empty';
    await page.click('#dreamRefresh');
    await page.waitForFunction(() => document.getElementById('dreamCycles').textContent.includes('No sleep cycles'));
    mode = 'unavailable';
    await page.click('#dreamRefresh');
    await page.waitForFunction(() => ['dreamPressure','dreamCycles','dreamScore'].every(id => document.getElementById(id).textContent.includes('unavailable')));
    assert(!await page.$('#dreamDetail [data-cycle]'));
    assert(requests.every(r => r.method === 'GET'), 'operator view issued a write');
    assert(requests.some(r => r.search.includes('before_id=dc-new')));
    console.log(JSON.stringify({passed:true,checks:['deep link','navigation','all sleep gates','cycle selection','compound cursor pagination','safe claim rendering','hidden tab quiet','single refresh listener','empty state','unavailable state','GET only'],requests:requests.length}));
  } finally { await browser.close(); }
})().catch(e => {console.error(e);process.exitCode=1;});

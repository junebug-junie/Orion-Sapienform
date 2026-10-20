// Retro play harness: drives a real browser through Hub, plays scripted input, and
// records what the player would feel. Diagnostic only; it changes no service.
//
//   node services/orion-hub/evals/retro_play.mjs --url https://athena.tail348bbe.ts.net \
//     --hub --cpu 4 --seconds 30 --out /tmp/retro-play [--js path/to/retro-panel.js]
//     [--evidence-dir ~/.local/share/orion/retro/service/evidence] [--game super-mario-bros]
//     [--override /static/js/retro-tab.js=services/orion-hub/static/js/retro-tab.js,...]
//
// --hub plays inside Hub's Retro tab (iframe, shares Hub's main thread); without it the
// standalone /retro page is used. --cpu throttles the renderer (CDP) to mimic a slower
// laptop. --js serves a local retro-panel.js in place of the deployed one, so client
// fixes can be tried against the live server without a deploy; --override generalizes it
// to any served path (comma-separated path=file pairs).
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(new URL('../package.json',import.meta.url));
const puppeteer=require('puppeteer');

const args=Object.fromEntries(process.argv.slice(2).reduce((acc,a,i,all)=>{
  if(a.startsWith('--')){const next=all[i+1];acc.push([a.slice(2),next===undefined||next.startsWith('--')?true:next]);}return acc;},[]));
// Explicit target only: a run starts (and replaces) a real game on that service.
if(!args.url||args.url===true){console.error('--url is required (e.g. https://athena.tail348bbe.ts.net)');process.exit(2);}
const url=String(args.url).replace(/\/$/,'');
const seconds=Number(args.seconds||30), cpu=Number(args.cpu||1), out=path.resolve(String(args.out||'retro-play-out'));
const game=args.game?String(args.game):null;
fs.mkdirSync(out,{recursive:true});
const pause=ms=>new Promise(r=>setTimeout(r,ms));

// Runs in every frame; records only inside the Retro page.
function instrument(){
  const rp=window.__rp={t0:performance.now(),sends:[],acks:[],rejects:[],status:[],longtasks:[],lag:[],closes:[]};
  try{new PerformanceObserver(l=>{for(const e of l.getEntries())rp.longtasks.push([Math.round(e.startTime),Math.round(e.duration)]);}).observe({type:'longtask',buffered:true});}catch(_){}
  let last=performance.now();setInterval(()=>{const now=performance.now();rp.lag.push(Math.round(now-last-50));last=now;},50);
  if(location.pathname!=='/retro')return;
  const sent=new Map(), Native=window.WebSocket;
  window.WebSocket=class extends Native{
    constructor(...a){super(...a);const input=String(a[0]).endsWith('/input');
      this.addEventListener('close',e=>rp.closes.push([Math.round(performance.now()),input?'input':'media',e.code]));
      if(input)this.addEventListener('message',e=>{let m;try{m=JSON.parse(e.data);}catch(_){return;}
        const now=performance.now();
        if(m.applied){const s=sent.get(m.applied.seq);if(s!==undefined)rp.acks.push([Math.round(now),Math.round(now-s)]);}
        else if(m.rejected||m.unknown)rp.rejects.push([Math.round(now),m.rejected||m.unknown,m.seq]);});}
    send(data){if(String(this.url).endsWith('/input')){try{const m=JSON.parse(data);sent.set(m.seq,performance.now());rp.sends.push([Math.round(performance.now()),m.seq,m.mask]);}catch(_){}}return super.send(data);}
  };
  document.addEventListener('DOMContentLoaded',()=>{const s=document.getElementById('status');if(!s)return;
    new MutationObserver(()=>rp.status.push([Math.round(performance.now()),s.textContent])).observe(s,{childList:true,characterData:true,subtree:true});});
}

const q=(xs,p)=>{if(!xs.length)return null;const s=[...xs].sort((a,b)=>a-b);return s[Math.min(s.length-1,Math.floor(p*s.length))];};
const stats=xs=>({n:xs.length,p50:q(xs,.5),p95:q(xs,.95),max:xs.length?Math.max(...xs):null});

const browser=await puppeteer.launch({headless:true,args:['--no-sandbox','--autoplay-policy=no-user-gesture-required']});
const report={url,hub:Boolean(args.hub),cpu,seconds,js:args.js||null,started:new Date().toISOString()};
let page;
try{
  page=await browser.newPage();await page.setViewport({width:1280,height:900});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.evaluateOnNewDocument(instrument);
  const overrides=new Map();
  if(args.js)overrides.set('/static/js/retro-panel.js',String(args.js));
  if(args.override)for(const pair of String(args.override).split(',')){const [k,v]=pair.split('=');overrides.set(k,v);}
  if(overrides.size){
    await page.setRequestInterception(true);
    page.on('request',r=>{const file=overrides.get(new URL(r.url()).pathname);
      if(file)r.respond({status:200,contentType:file.endsWith('.html')?'text/html':'application/javascript',headers:{'Cache-Control':'no-store'},body:fs.readFileSync(file)});
      else r.continue();});
  }
  report.overrides=[...overrides.keys()];
  if(cpu>1){const cdp=await page.createCDPSession();await cdp.send('Emulation.setCPUThrottlingRate',{rate:cpu});}
  let frame;
  if(args.hub){
    await page.goto(url+'/#retro',{waitUntil:'domcontentloaded',timeout:60000});
    await page.waitForFunction(()=>document.querySelector('#retroPanelFrame')?.getAttribute('src')==='/retro',{timeout:30000});
    frame=await (await page.$('#retroPanelFrame')).contentFrame();
  } else {await page.goto(url+'/retro',{waitUntil:'domcontentloaded'});frame=page.mainFrame();}
  await frame.waitForSelector('#player:not([hidden])',{timeout:30000});
  if(game)await frame.select('#game',game);
  report.game=await frame.$eval('#game',e=>e.value);
  await frame.click('#new');
  await frame.waitForFunction(()=>Number(document.querySelector('#screen').dataset.appliedFrame)>0,{timeout:15000}).catch(()=>{});
  await frame.click('#screen');await page.keyboard.press('Enter');
  // Scripted play: run right, jump every ~0.7 s, release briefly every few seconds.
  const end=Date.now()+seconds*1000;let i=0;
  while(Date.now()<end){
    await page.keyboard.down('ArrowRight');await page.keyboard.down('KeyZ');await pause(250);await page.keyboard.up('KeyZ');await pause(450);
    if(++i%6===0){await page.keyboard.up('ArrowRight');await pause(300);}
  }
  await page.keyboard.up('ArrowRight');
  const rp=await frame.evaluate(()=>window.__rp), hubRp=args.hub?await page.evaluate(()=>window.__rp):null;
  const session=await frame.evaluate(async()=>(await (await fetch('/api/retro/health/ready')).json()).session);
  await frame.click('#more summary').catch(()=>{});await frame.click('#pause').catch(()=>{});
  const gaps=rp.sends.slice(1).map((s,k)=>s[0]-rp.sends[k][0]);
  report.client={
    send_interval_ms:stats(gaps), ack_ms:stats(rp.acks.map(a=>a[1])),
    sends:rp.sends.length, acks:rp.acks.length, rejects:rp.rejects, closes:rp.closes,
    status:rp.status.map(([t,s])=>[t,s]), longtask_ms_total:rp.longtasks.reduce((a,b)=>a+b[1],0),
    longtasks_over_100ms:rp.longtasks.filter(l=>l[1]>100).length, timer_lag_ms:stats(rp.lag),
    hub_longtask_ms_total:hubRp?hubRp.longtasks.reduce((a,b)=>a+b[1],0):null,
    frames_drawn:await frame.$eval('#screen',c=>Number(c.dataset.frame)||0), errors};
  report.session={id:session?.session_id,state:session?.state,frame:session?.frame};
  if(args['evidence-dir'] && session?.session_id){
    const file=path.join(String(args['evidence-dir']).replace(/^~/,process.env.HOME),session.session_id+'.jsonl');
    if(fs.existsSync(file)){
      const rows=fs.readFileSync(file,'utf8').trim().split('\n').map(l=>JSON.parse(l));
      const ts=r=>Date.parse(r.utc), applied=rows.filter(r=>r.kind==='applied');
      report.server={applied:applied.length,
        applied_gap_ms:stats(applied.slice(1).map((r,k)=>ts(r)-ts(applied[k]))),
        events:rows.filter(r=>r.kind!=='applied').map(r=>[r.utc.slice(11,23),r.kind,r.state||r.action||'',r.reason||''])};
    }
  }
  await page.screenshot({path:path.join(out,'screen.png')});
} catch(error){report.error=String(error&&error.stack||error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}
finally{await browser.close();}
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
const c=report.client||{};
console.log(JSON.stringify({error:report.error?.split('\n')[0],game:report.game,cpu,hub:report.hub,
  send_interval_ms:c.send_interval_ms,ack_ms:c.ack_ms,timer_lag_ms:c.timer_lag_ms,longtask_ms_total:c.longtask_ms_total,
  hub_longtask_ms_total:c.hub_longtask_ms_total,rejects:c.rejects?.length,closes:c.closes?.length,
  last_status:c.status?.at(-1)?.[1],frames_drawn:c.frames_drawn,server:report.server&&{applied:report.server.applied,gap:report.server.applied_gap_ms,
  drops:report.server.events.filter(e=>/disconnect|no_controller/.test(e[3])).length}},null,1));

// Zelda co-play harness: a real browser loads a save, hands the controller to Orion
// ("Try Orion goals") and hands it back each time a 60 s grant ends, then reads what
// Orion did from Retro's journal and orion-prediction's action bets. Diagnostic only.
//
//   node services/orion-hub/evals/retro_coplay.mjs --url https://athena.tail348bbe.ts.net \
//     --game the-legend-of-zelda-prg1 [--save <save_id>] --minutes 5 --out /tmp/coplay \
//     [--prediction http://127.0.0.1:8132]
//
// Like retro_play.mjs, a run starts (loads) a real game and ends any game already running.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(new URL('../package.json',import.meta.url));
const puppeteer=require(process.env.RETRO_PUPPETEER_PACKAGE?createRequire(process.env.RETRO_PUPPETEER_PACKAGE).resolve('puppeteer'):'puppeteer');

const args=Object.fromEntries(process.argv.slice(2).reduce((acc,a,i,all)=>{
  if(a.startsWith('--'))acc.push([a.slice(2),all[i+1]&&!all[i+1].startsWith('--')?all[i+1]:true]);return acc;},[]));
const url=String(args.url||''), game=String(args.game||'the-legend-of-zelda-prg1');
const minutes=Number(args.minutes||5), out=String(args.out||'/tmp/coplay'), prediction=String(args.prediction||'http://127.0.0.1:8132');
if(!url){console.error('--url is required');process.exit(2);}
fs.mkdirSync(out,{recursive:true});
const pause=ms=>new Promise(r=>setTimeout(r,ms));

const browser=await puppeteer.launch({headless:true,args:['--no-sandbox','--autoplay-policy=no-user-gesture-required']});
const report={url,game,minutes,started:new Date().toISOString(),handoffs:[],policy:[]};
let page;
try{
  page=await browser.newPage();await page.setViewport({width:1280,height:900});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));report.errors=errors;
  // Record every socket close: who closed it (stack) or the server's code/reason.
  await page.evaluateOnNewDocument(()=>{
    window.__closes=[];const W=window.WebSocket;
    window.WebSocket=class extends W{constructor(...a){super(...a);
      this.addEventListener('close',e=>window.__closes.push({t:Math.round(performance.now()),url:String(this.url).split('/').pop(),code:e.code,reason:e.reason,clean:e.wasClean}));}
      close(...a){window.__closes.push({t:Math.round(performance.now()),url:String(this.url).split('/').pop(),by:'page',stack:new Error().stack.split('\n').slice(2,6).join(' | ')});return super.close(...a);}};
    for(const ev of ['blur','focus','pagehide'])window.addEventListener(ev,()=>window.__closes.push({t:Math.round(performance.now()),event:ev}));
    document.addEventListener('visibilitychange',()=>window.__closes.push({t:Math.round(performance.now()),event:'visibility',hidden:document.hidden}));
  });
  await page.goto(url+'/retro',{waitUntil:'domcontentloaded',timeout:60000});
  await page.waitForSelector('#player:not([hidden])',{timeout:30000});
  await page.select('#game',game);
  await page.waitForFunction(()=>document.querySelector('#saves').options.length>0,{timeout:15000});
  if(args.save)await page.select('#saves',String(args.save));
  report.save=await page.$eval('#saves',e=>e.value);
  await page.click('#load');
  await page.waitForFunction(()=>Number(document.querySelector('#screen').dataset.appliedFrame)>0,{timeout:20000}).catch(()=>{});
  await pause(1500);
  // Like a person: click the game to connect this controller, then open More.
  if(!args['no-click'])await page.click('#screen');
  await page.waitForFunction(()=>/Playing|Connected/.test(document.querySelector('#status')?.textContent||''),{timeout:15000}).catch(()=>{});
  report.status_before=await page.$eval('#status',e=>e.textContent.trim()).catch(()=>null);
  await page.click('#more summary').catch(()=>{});
  const end=Date.now()+minutes*60000;
  let last='';
  while(Date.now()<end){
    const text=await page.$eval('#policy',e=>e.textContent.trim());
    if(text!==last){report.policy.push([new Date().toISOString().slice(11,19),text.slice(0,240)]);last=text;}
    if(text==='No scripted controller active.'){
      if(report.handoffs.length)await page.click('#screen');   // the grant ended: take the game back first
      await pause(800);
      await page.click('#cognitive');report.handoffs.push(new Date().toISOString());
      await pause(3000);
      report.status=(report.status||[]).concat([[new Date().toISOString().slice(11,19),await page.$eval('#status',e=>e.textContent.trim()).catch(()=>null)]]).slice(-40);
    }
    await pause(1000);
  }
  report.closes=await page.evaluate(()=>window.__closes);
  report.session=await page.evaluate(async()=>(await (await fetch('/api/retro/health/ready')).json()).session);
  await page.screenshot({path:path.join(out,'screen.png')});
  await page.click('#pause').catch(()=>{});
} catch(error){report.error=String(error&&error.stack||error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}
finally{await browser.close();}
const sid=report.session?.session_id;
if(sid){
  try{report.bets=(await (await fetch(`${prediction}/action-bets?session_id=${sid}`)).json()).bets||[];}
  catch(e){report.bets_error=String(e);}
}
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
const bets=report.bets||[];
const by=k=>bets.reduce((m,b)=>(m[b[k]??'open']=(m[b[k]??'open']||0)+1,m),{});
console.log(JSON.stringify({error:report.error?.split('\n')[0],session:sid,save:report.save,handoffs:report.handoffs.length,
  page_errors:report.errors?.length,bets:bets.length,results:by('result'),skills:by('skill'),
  ranks:by('chosen_rank'),last_policy:report.policy.at(-1)?.[1]},null,1));

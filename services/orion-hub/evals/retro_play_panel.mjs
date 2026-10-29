// Headless check of Orion's play panel. Driven by evals/retro_play_panel.py (fixture Retro).
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(process.env.RETRO_PUPPETEER_PACKAGE || new URL('../package.json',import.meta.url));
const puppeteer=require('puppeteer');
const out=process.env.RETRO_EVAL_OUTPUT,url=process.env.RETRO_EVAL_URL;
const browser=await puppeteer.launch({headless:true,args:['--no-sandbox']});
const fixture=async()=>(await fetch(url+'/__fixture/log')).json();
try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const assets=[];page.on('response',r=>{if(r.url().includes('/static/js/'))assets.push([r.url(),r.status()]);});
  await page.goto(url+'/retro');
  // Rendered template + linked asset.
  assert(assets.some(([u,s])=>u.includes('/static/js/retro-panel.js?v=13') && s===200),JSON.stringify(assets));
  // Leisure is visible live on open, by watching only.
  await page.waitForSelector('#play:not([hidden])',{timeout:8000});
  await page.waitForFunction(()=>document.querySelector('#leisure').textContent.includes('playing alone'),{timeout:8000});
  await page.waitForFunction(()=>document.querySelectorAll('#atlas .room').length===128,{timeout:10000});
  let log=await fixture();
  assert(log.retro.some(r=>r.op==='ws' && r.channel==='media'),'watching never opened the media socket');
  assert(!log.retro.some(r=>r.op==='ws' && r.channel==='input'),'watching must not open an input socket');
  assert(!log.retro.some(r=>r.op==='control'),'watching must not take a controller');
  assert.equal(log.human,0,'opening the page is not a human Retro action');
  // Atlas: known rooms, current room, used exits, caves, deaths.
  const atlas=await page.evaluate(()=>{const q=s=>[...document.querySelectorAll(s)].map(e=>Number(e.dataset.room));
    return {known:q('#atlas .room.known'),current:q('#atlas .room.current'),north:q('#atlas .room.xN'),
      east:q('#atlas .room.xE'),death:q('#atlas .room.death'),
      cave:document.querySelector('#atlas .room[data-room="119"]').textContent,
      deaths:document.querySelector('#atlas .room[data-room="120"]').textContent,
      info:document.querySelector('#atlasInfo').textContent};});
  assert.deepEqual(atlas.known.sort((a,b)=>a-b),[103,119,120],JSON.stringify(atlas));
  assert.deepEqual(atlas.current,[119]);assert(atlas.north.includes(119) && atlas.east.includes(119));
  assert.deepEqual(atlas.death,[120]);assert.equal(atlas.cave,'c');assert.equal(atlas.deaths,'×2');
  assert(atlas.info.startsWith('3 rooms known'),atlas.info);
  // Goal, rationale, prediction, verdict history.
  await page.waitForFunction(()=>document.querySelectorAll('#verdicts li').length===5,{timeout:5000});
  const goal=await page.evaluate(()=>({goal:document.querySelector('#zgoal').textContent,
    prediction:document.querySelector('#zprediction').textContent,
    verdicts:[...document.querySelectorAll('#verdicts li')].map(li=>[li.className,li.textContent])}));
  assert(goal.goal.includes('exit N') && goal.goal.includes('North of the start is unexplored'),goal.goal);
  assert(goal.prediction.includes('reach_room 103'),goal.prediction);
  assert(goal.verdicts.some(([c])=>c==='met') && goal.verdicts.some(([c])=>c==='refuted'),JSON.stringify(goal.verdicts));
  await page.screenshot({path:out+'/play-panel.png',fullPage:true});
  // Heat shed: the page says why Orion is not choosing.
  await fetch(url+'/__fixture/waiting',{method:'POST'});
  await page.waitForFunction(()=>document.querySelector('#leisure').textContent.startsWith('Orion is waiting: cabinet hot'),{timeout:5000});
  assert(await page.$eval('#leisure',e=>e.classList.contains('waiting')));
  // Takeover stays intact and asks Retro for a controller (Retro's arbiter preempts leisure).
  await page.$eval('#more',d=>{d.open=true;});
  await page.click('#takeover');
  for(let i=0;i<50;i++){log=await fixture();if(log.retro.some(r=>r.op==='control'))break;await new Promise(r=>setTimeout(r,100));}
  const control=log.retro.find(r=>r.op==='control');
  assert(control && control.body.action==='acquire' && control.body.port===0,JSON.stringify(log.retro));
  assert(log.retro.some(r=>r.op==='ws' && r.channel==='input'),'takeover never opened the input socket');
  assert(log.human>=1,'the relay did not record the human action');
  // Handing off to Orion survives the window losing focus: the sponsor socket stays open,
  // because Retro ends a grant whose sponsor disconnects.
  const closed=l=>l.retro.filter(r=>r.op==='ws_closed' && r.channel==='input').length;
  const before=closed(log);
  await page.click('#cognitive');
  for(let i=0;i<50;i++){log=await fixture();if(log.retro.some(r=>r.op==='scripted'))break;await new Promise(r=>setTimeout(r,100));}
  assert(log.retro.some(r=>r.op==='scripted' && r.body.cognition===true),JSON.stringify(log.retro));
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));
  await new Promise(r=>setTimeout(r,1500));
  log=await fixture();
  assert.equal(closed(log),before,'a blur during a handoff closed the sponsor socket and would end Orion\'s turn');
  // Once that grant ends while the game runs on, the page leaves handoff mode: a blur releases the socket.
  await fetch(url+'/__fixture/end_grant',{method:'POST'});
  await page.waitForFunction(()=>document.querySelector('#policy').textContent==='No scripted controller active.',{timeout:5000});
  await new Promise(r=>setTimeout(r,300));
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));
  for(let i=0;i<50;i++){log=await fixture();if(closed(log)>before)break;await new Promise(r=>setTimeout(r,100));}
  assert(closed(log)>before,'after the handoff ended, a blur must release the socket (page stuck in handoff mode)');
  // Let Orion play: a 30-minute hand-over, renewed when the grant ends, until the player clicks the game.
  const starts=l=>l.retro.filter(r=>r.op==='scripted' && r.body?.action==='start');
  let n0=starts(await fixture()).length;
  await page.evaluate(()=>window.dispatchEvent(new Event('focus')));
  await page.click('#orionplay');
  for(let i=0;i<50;i++){log=await fixture();if(starts(log).length>n0)break;await new Promise(r=>setTimeout(r,100));}
  const first=starts(log).at(-1);
  assert(first && first.body.duration_seconds===1800 && first.body.cognition===true,JSON.stringify(starts(log)));
  await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Orion is playing'),{timeout:5000});
  await new Promise(r=>setTimeout(r,1500));                        // the page sees its grant in a status
  await fetch(url+'/__fixture/end_grant?expired=1',{method:'POST'});   // the turn ran out: the page renews it
  n0=starts(log).length;
  for(let i=0;i<80;i++){log=await fixture();if(starts(log).length>n0)break;await new Promise(r=>setTimeout(r,100));}
  assert(starts(log).length>n0 && starts(log).at(-1).body.duration_seconds===1800,'the ended turn was not renewed');
  const controls=l=>l.retro.filter(r=>r.op==='control').length;
  const c0=controls(log);
  await page.waitForFunction(()=>document.querySelector('#orionplay').getAttribute('aria-pressed')==='true',{timeout:3000});
  await new Promise(r=>setTimeout(r,1500));
  await page.click('#screen');                                      // the player takes the game back
  for(let i=0;i<50;i++){log=await fixture();if(controls(log)>c0)break;await new Promise(r=>setTimeout(r,100));}
  assert(controls(log)>c0,'a click while Orion played did not take the controller back');
  n0=starts(log).length;
  await fetch(url+'/__fixture/end_grant',{method:'POST'});
  await new Promise(r=>setTimeout(r,2500));
  log=await fixture();
  assert.equal(starts(log).length,n0,'after a take-back, Orion must not get the controller again');
  // Someone else takes the controller mid-turn: Orion stops instead of taking it back (review H1).
  await page.click('#orionplay');
  for(let i=0;i<50;i++){log=await fixture();if(starts(log).length>n0)break;await new Promise(r=>setTimeout(r,100));}
  await new Promise(r=>setTimeout(r,1500));
  n0=starts(log).length;
  await fetch(url+'/__fixture/end_grant',{method:'POST'});          // not expired: another controller took it
  await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Someone else took the controller'),{timeout:5000});
  await new Promise(r=>setTimeout(r,1500));
  log=await fixture();
  assert.equal(starts(log).length,n0,'Orion took the controller back from someone else');
  assert.equal(await page.$eval('#orionplay',b=>b.getAttribute('aria-pressed')),'false');
  assert.deepEqual(errors,[]);
  fs.writeFileSync(out+'/play-panel.json',JSON.stringify({atlas,goal,retro:log.retro.map(r=>r.op+(r.channel?':'+r.channel:'')),human:log.human},null,2));
}finally{await browser.close();}

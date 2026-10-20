import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require=createRequire(process.env.RETRO_PUPPETEER_PACKAGE || new URL('../package.json',import.meta.url));
const puppeteer=require('puppeteer');
const out=process.env.RETRO_EVAL_OUTPUT,url=process.env.RETRO_EVAL_URL;
const browser=await puppeteer.launch({headless:true,args:['--no-sandbox','--autoplay-policy=no-user-gesture-required']});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
let page;
try{
  page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.evaluateOnNewDocument(delay=>{
    if(delay){
      const NativeSocket=window.WebSocket;
      window.WebSocket=class extends NativeSocket {
        set onmessage(callback){super.onmessage=event=>{
          const slow=this.url.endsWith('/input') && JSON.parse(event.data).applied;
          if(slow && window.__dropRetroAck)return;
          if(slow)setTimeout(()=>callback(event),delay);else callback(event);
        };}
      };
    }

    window.__audioPeak=0;
    const original=AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start=function(...args){
      if(this.buffer){const x=this.buffer.getChannelData(0);for(const v of x)window.__audioPeak=Math.max(window.__audioPeak,Math.abs(v));}
      return original.apply(this,args);
    };
  },Number(process.env.RETRO_ACK_DELAY_MS || 0));
  await page.goto(url+'/retro');
  assert.equal(await page.evaluate(async()=> (await fetch('/api/retro/games')).status),200);
  await page.waitForSelector('#player:not([hidden])');await page.$eval('#more',d=>{d.open=true;});
  if(await page.$('#code')) throw new Error('Unexpected access-code prompt');
  await page.select('#game',process.env.RETRO_EVAL_GAME);await page.click('#new');
  await page.waitForFunction(()=>Number(document.querySelector('#screen').dataset.appliedFrame)>10,{timeout:10000});
  await page.click('#start');await pause(600);await page.click('#screen');
  await page.keyboard.down('ArrowRight');await page.keyboard.down('z');
  await pause(Number(process.env.RETRO_EVAL_SECONDS)*1000);
  await page.keyboard.up('ArrowRight');await page.keyboard.up('z');
  const observed=await page.evaluate(()=>{
    const c=document.querySelector('#screen'),pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
    const colors=new Set();for(let i=0;i<pixels.length;i+=4)colors.add((pixels[i]<<16)|(pixels[i+1]<<8)|pixels[i+2]);
    return {frame:Number(c.dataset.frame),applied:Number(c.dataset.appliedFrame),audioFrames:Number(c.dataset.audioFrames),audioPeak:window.__audioPeak,colors:colors.size};
  });
  assert(observed.frame>60 && observed.applied>60 && observed.audioFrames>48000 && observed.audioPeak>0 && observed.colors>=2,JSON.stringify(observed));
  await page.screenshot({path:out+'/browser.png',fullPage:true});
  await page.click('#join');await pause(500);
  // Verify gamepad mapping through the actual authority's applied-action journal.
  await page.evaluate(()=>{
    const buttons=Array.from({length:16},()=>({pressed:false}));buttons[0].pressed=true;buttons[15].pressed=true;
    Object.defineProperty(navigator,'getGamepads',{configurable:true,value:()=>[{mapping:'standard',buttons,axes:[0,0]}]});
  });
  await pause(250);
  const state=await page.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session);
  const actions=await page.evaluate(async sid=>(await (await fetch(`/api/retro/sessions/${sid}/evidence`)).json()).applied,state.session_id);
  assert(actions.some(a=>a.mask===384),'gamepad right+A never applied');
  await page.select('#port','1');await pause(150);await page.click('#join');await pause(400);
  await page.click('#takeover');await pause(250);
  const p2actions=await page.evaluate(async sid=>(await (await fetch(`/api/retro/sessions/${sid}/evidence`)).json()).applied,state.session_id);
  assert(p2actions.some(a=>a.port===1 && a.mask===384),'P2 gamepad input never applied');
  if(Number(process.env.RETRO_ACK_DELAY_MS || 0)){
    // Lost acknowledgments still release control; restored ones resume play with no Connect click.
    await page.evaluate(()=>{window.__dropRetroAck=true;});
    await page.waitForFunction(()=>/Reconnecting|Could not/.test(document.querySelector('#status').textContent),{timeout:3000});
    await page.evaluate(()=>{window.__dropRetroAck=false;});
    // Acks delayed from before the drop can still land; measure only after the heal's new lease.
    await page.waitForFunction(async()=>(await (await fetch('/api/retro/health/ready')).json()).session.state==='running',{timeout:6000,polling:100});
    const before=await page.$eval('#screen',c=>Number(c.dataset.appliedFrame));
    await page.waitForFunction(b=>Number(document.querySelector('#screen').dataset.appliedFrame)>b+30,{timeout:6000},before);
  }
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));await pause(400);
  const afterBlur=await page.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state);
  assert.equal(afterBlur,'paused');
  await page.click('#join');await pause(500);
  const resumed=await page.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state);
  assert.equal(resumed,'running');
  // Real native script handoff, visible identity, human takeover and disconnect.
  await page.evaluate(()=>Object.defineProperty(navigator,'getGamepads',{configurable:true,value:()=>[]}));
  await page.click('#scripted');
  await page.waitForFunction(()=>document.querySelector('#policy').textContent.includes('scripted-right-jump-v1'));
  await pause(600);
  const scriptedActions=await page.evaluate(async sid=>(await (await fetch(`/api/retro/sessions/${sid}/evidence`)).json()).applied,state.session_id);
  assert(scriptedActions.some(a=>a.policy_id==='scripted-right-jump-v1' && a.mask===384),'script never actuated');
  await page.click('#takeover');
  await page.waitForFunction(()=>document.activeElement===document.querySelector('#screen'));
  await page.keyboard.down('ArrowLeft');await pause(250);await page.keyboard.up('ArrowLeft');
  await page.waitForFunction(async()=>(await (await fetch('/api/retro/health/ready')).json()).session.scripted.length===0);
  const humanActions=await page.evaluate(async sid=>(await (await fetch(`/api/retro/sessions/${sid}/evidence`)).json()).applied,state.session_id);
  assert(humanActions.some(a=>!a.policy_id && a.mask===64),'human takeover never actuated');
  // Synthetic planner, real native goal mailbox, input attribution and takeover.
  // This is UI/boundary coverage; cognitive_loop.py separately requires a real LLM.
  await page.click('#cognitive');
  await page.waitForFunction(()=>document.querySelector('#policy').textContent.includes('goal-motor-v1'));
  await page.waitForFunction(async()=>{
    const s=(await (await fetch('/api/retro/health/ready')).json()).session;
    return s.scripted.some(p=>p.cognition?.goal?.planner_id==='browser-fixture');
  });
  const cognitiveActions=await page.evaluate(async sid=>(await (await fetch(`/api/retro/sessions/${sid}/evidence`)).json()).applied,state.session_id);
  assert(cognitiveActions.some(a=>a.policy_id==='goal-motor-v1' && a.goal_attempt_id && a.mask===0),'acknowledged wait never applied');
  await page.click('#takeover');
  await page.waitForFunction(()=>document.activeElement===document.querySelector('#screen'));
  await page.keyboard.down('ArrowLeft');await pause(200);await page.keyboard.up('ArrowLeft');
  await page.click('#scripted');await pause(300);
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));await pause(400);
  const stoppedScript=await page.evaluate(async()=>(await (await fetch('/api/retro/health/ready')).json()).session);
  assert.equal(stoppedScript.state,'paused');assert.deepEqual(stoppedScript.scripted,[]);
  await page.click('#screen');
  await page.waitForFunction(async()=>(await (await fetch('/api/retro/health/ready')).json()).session.state==='running');
  await pause(300);
  // A rejected handoff cannot strand the human's existing connection.
  await page.evaluate(()=>{
    const original=window.fetch;
    window.fetch=(path,opts)=>{
      if(String(path).endsWith('/scripted')){
        window.fetch=original;
        opts={...opts,body:JSON.stringify({...JSON.parse(opts.body),expected_epoch:999999})};
      }
      return original(path,opts);
    };
  });
  await page.click('#scripted');
  await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Handoff failed'));
  await page.click('#takeover');
  await page.waitForFunction(()=>document.activeElement===document.querySelector('#screen'));
  await page.keyboard.down('ArrowLeft');await pause(250);await page.keyboard.up('ArrowLeft');
  const recovered=await page.evaluate(async sid=>(await (await fetch(`/api/retro/sessions/${sid}/evidence`)).json()).applied,state.session_id);
  assert(recovered.slice(-8).some(a=>!a.policy_id && a.mask===64),'rejected handoff stranded human');
  // Exercise server-enforced budget with the same UI; shorten only this request.
  await page.evaluate(()=>{
    const original=window.fetch;
    window.fetch=(path,opts)=>{
      if(String(path).endsWith('/scripted')){
        window.fetch=original;
        opts={...opts,body:JSON.stringify({...JSON.parse(opts.body),duration_seconds:5})};
      }
      return original(path,opts);
    };
  });
  await page.click('#scripted');
  await page.waitForFunction(async()=>{
    const s=(await (await fetch('/api/retro/health/ready')).json()).session;
    return s.state==='paused' && s.scripted.length===0;
  },{timeout:8000,polling:100});
  await page.click('#takeover');
  await page.waitForFunction(async()=>(await (await fetch('/api/retro/health/ready')).json()).session.state==='running');
  await pause(300);
  await page.click('#scripted');await pause(200);await page.click('#pause');await pause(200);
  await page.click('#screen');
  await page.waitForFunction(async()=>(await (await fetch('/api/retro/health/ready')).json()).session.state==='running');
  await pause(300);
  await page.click('#pause');await pause(150);await page.click('#join');await pause(500);
  // Cancel an in-flight resume: it must not acquire control after blur.
  await page.click('#pause');await pause(150);
  await page.evaluate(()=>{
    const original=window.fetch;window.__resumePending=false;
    window.fetch=async(...args)=>{const r=await original(...args);if(String(args[0]).endsWith('/resume')){window.__resumePending=true;await new Promise(r=>setTimeout(r,300));}return r;};
  });
  await page.click('#join');
  await page.waitForFunction(()=>window.__resumePending);
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));
  await pause(5700);
  assert.equal(await page.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state),'paused');
  await page.click('#stop');await pause(150);
  assert.equal(await page.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state),'stopped');
  const tab=await browser.newPage();await tab.goto(url+'/__retro_tab_test');
  await tab.click('#retroTabButton');
  await tab.waitForFunction(()=>document.querySelector('#retroPanelFrame').getAttribute('src')==='/retro');
  let frame=await (await tab.$('#retroPanelFrame')).contentFrame();
  await frame.waitForSelector('#player:not([hidden])');await frame.$eval('#more',d=>{d.open=true;});await frame.select('#game',process.env.RETRO_EVAL_GAME);
  await frame.click('#new');await frame.waitForFunction(()=>Number(document.querySelector('#screen').dataset.appliedFrame)>10);
  await tab.click('#hubTabButton');await pause(400);
  assert.equal(await tab.$eval('#retroPanelFrame',e=>e.getAttribute('src')),'about:blank');
  assert.equal(await tab.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state),'paused');
  await tab.click('#retroTabButton');await pause(300);frame=await (await tab.$('#retroPanelFrame')).contentFrame();
  await frame.waitForSelector('#player:not([hidden])');await frame.$eval('#more',d=>{d.open=true;});await frame.click('#join');await pause(400);
  assert.equal(await tab.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state),'running');
  // Another self-routing tab can hide Retro before its hash handler executes.
  await tab.evaluate(()=>{document.querySelector('#retro').classList.add('hidden');location.hash='#personality';});
  await tab.waitForFunction(()=>document.querySelector('#retroPanelFrame').getAttribute('src')==='about:blank');await pause(400);
  assert.equal(await tab.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state),'paused');
  await tab.click('#retroTabButton');await pause(300);frame=await (await tab.$('#retroPanelFrame')).contentFrame();
  await frame.waitForSelector('#player:not([hidden])');await frame.$eval('#more',d=>{d.open=true;});await frame.click('#stop');
  await pause(150);assert.equal(await tab.evaluate(async()=> (await (await fetch('/api/retro/health/ready')).json()).session.state),'stopped');
  await tab.close();
  assert.deepEqual(errors,[]);
  fs.writeFileSync(out+'/browser.json',JSON.stringify({status:'PASS',ackDelayMs:Number(process.env.RETRO_ACK_DELAY_MS || 0),game:process.env.RETRO_EVAL_GAME,observed,cognitiveFixtureHandoff:true,scriptedHandoff:true,scriptedTakeover:true,scriptedDisconnect:true,hubTab:true,tabExitPaused:true,keyboard:true,syntheticStandardGamepad:true,blurPaused:true,reconnect:true,selfHeal:Boolean(Number(process.env.RETRO_ACK_DELAY_MS || 0)),unverified:['physical gamepad','user-device audio output','input-to-photon latency','30-minute actual-client A/V drift','cognition/learning']},null,2));
} catch(error) { if(page){console.error(await page.$eval('#status',e=>e.textContent));await page.screenshot({path:out+'/failure.png'});}throw error; } finally {await browser.close();}

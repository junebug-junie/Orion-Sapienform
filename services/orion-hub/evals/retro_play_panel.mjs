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
  await page.evaluateOnNewDocument(()=>{
    Object.defineProperty(navigator,'getGamepads',{configurable:true,value:()=>[]});
  });
  const assets=[];page.on('response',r=>{if(r.url().includes('/static/js/'))assets.push([r.url(),r.status()]);});
  await page.goto(url+'/retro');
  // Rendered template + linked asset.
  assert(assets.some(([u,s])=>u.includes('/static/js/retro-panel.js?v=20') && s===200),JSON.stringify(assets));
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
  // A media-only drop must recover without closing/reacquiring the input sponsor.
  await page.waitForFunction(()=>Number(document.querySelector('#screen').dataset.frame)>0);
  const mediaBefore=log.retro.filter(r=>r.op==='ws' && r.channel==='media').length;
  const controlsBefore=log.retro.filter(r=>r.op==='control').length;
  await fetch(url+'/__fixture/media_drop',{method:'POST'});
  for(let i=0;i<30;i++){
    log=await fixture();
    if(log.retro.filter(r=>r.op==='ws' && r.channel==='media').length>mediaBefore)break;
    await new Promise(r=>setTimeout(r,100));
  }
  assert(log.retro.filter(r=>r.op==='ws' && r.channel==='media').length>mediaBefore,'media did not reconnect');
  assert.equal(closed(log),before,'media failure closed input');
  assert.equal(log.retro.filter(r=>r.op==='control').length,controlsBefore,'media recovery reacquired control');
  // A socket that stays open without pictures is also a failure: held input goes neutral.
  await page.keyboard.down('ArrowRight');
  await fetch(url+'/__fixture/media_silent',{method:'POST'});
  await new Promise(r=>setTimeout(r,1400));log=await fixture();
  assert.equal(log.retro.filter(r=>r.op==='input').at(-1)?.mask,0,'frozen picture kept held input');
  await page.keyboard.up('ArrowRight');
  await fetch(url+'/__fixture/media_silent?silent=false',{method:'POST'});
  await new Promise(r=>setTimeout(r,300));
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
  // The sitting record (game company design, patch 1): a button press opens a sitting (the
  // take-backs above may have opened it already); the page asks who plays, once; the answer
  // reaches the server; the game keeps the keyboard.
  const sittingNow=async()=>(await (await fetch(url+'/api/retro/sitting')).json()).sitting;
  await page.click('#screen');                                      // take the controller
  let opened=await sittingNow();
  for(let i=0;i<40 && !opened;i++){                                 // press until the relay saw a change
    await page.keyboard.down('ArrowRight');await new Promise(r=>setTimeout(r,120));
    await page.keyboard.up('ArrowRight');await new Promise(r=>setTimeout(r,130));
    opened=await sittingNow();
  }
  assert(opened && opened.player_stated===null && opened.mask_changes>=1,'a key press did not open a sitting: '+JSON.stringify(opened));
  await page.waitForFunction(()=>!document.querySelector('#whoPlays').hidden,{timeout:9000});
  assert.equal(await page.$eval('#stage',s=>s.contains(document.querySelector('#whoPlays'))),true); // visible in fullscreen
  // "Not now" is an answer: the box goes, the server still knows that nobody said, and neither the
  // next poll nor a reload asks again.
  await page.click('#whoLater');
  await page.waitForFunction(()=>document.querySelector('#whoPlays').hidden,{timeout:3000});
  assert.equal((await sittingNow()).player_stated,null);
  const ready=async()=>{await page.waitForSelector('#play:not([hidden])',{timeout:8000});await page.click('#screen');};
  const press=async()=>{await page.keyboard.down('ArrowLeft');await new Promise(r=>setTimeout(r,120));await page.keyboard.up('ArrowLeft');};
  await page.reload();await ready();await press();
  await new Promise(r=>setTimeout(r,6500));
  assert.equal(await page.$eval('#whoPlays',e=>e.hidden),true,'the question came back after "Not now" and a reload');
  // A page with no memory of the question asks, but only once it holds the controller: a second
  // screen that only watches is never asked.
  await page.evaluate(()=>sessionStorage.removeItem('retroSittingAsked'));
  await page.reload();await page.waitForSelector('#play:not([hidden])',{timeout:8000});
  await new Promise(r=>setTimeout(r,6500));
  assert.equal(await page.$eval('#whoPlays',e=>e.hidden),true,'a page that only watches was asked who plays');
  await page.click('#screen');await press();
  await page.waitForFunction(()=>!document.querySelector('#whoPlays').hidden,{timeout:15000});
  await page.click('#whoGuest');
  await page.waitForFunction(()=>document.querySelector('#whoPlays').hidden,{timeout:3000});
  assert.equal((await sittingNow()).player_stated,'guest');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'screen','the game lost the keyboard after the answer');
  await new Promise(r=>setTimeout(r,5500));                         // one more poll: asked once, not again
  assert.equal(await page.$eval('#whoPlays',e=>e.hidden),true);
  // Orion's look (game company patch 2): the page shows the server's own reason, never its own words.
  // This harness has no database, so no game has a calibrated floor: the page must say so, not "watching".
  await page.waitForFunction(()=>!document.querySelector('#watch').hidden,{timeout:9000});
  const watchText=await page.$eval('#watchText',e=>e.textContent);
  assert(watchText.startsWith('No looks for this game yet'),'watch line: '+watchText);
  assert.equal(await page.$eval('#watchSwitch',e=>e.value),'on',"Juniper's switch starts on (watch and talk)");
  // Three positions (patch 3): Off / Watch quietly / Watch and talk map to off / quiet / on.
  assert.deepEqual(await page.$$eval('#watchSwitch option',o=>o.map(x=>[x.value,x.textContent])),
    [['off','Off'],['quiet','Watch quietly'],['on','Watch and talk']]);
  const watchPost=page.waitForRequest(r=>r.url().endsWith('/api/retro/sitting/watch') && r.method()==='POST',{timeout:3000});
  await page.select('#watchSwitch','quiet');
  assert.deepEqual(JSON.parse((await watchPost).postData()),{mode:'quiet'});
  await page.waitForFunction(()=>!document.querySelector('#watchSwitch').disabled,{timeout:3000});
  const watchAfter=await page.$eval('#watchSwitch',e=>e.value);   // stored, or put back with the server's reason
  assert(['quiet','on'].includes(watchAfter),watchAfter);
  await page.click('#screen');
  // --- The chat box (game company patch 3) ---------------------------------------------------------
  const chatFix=async()=>(await fetch(url+'/__fixture/chat')).json();
  const sittingId=(await sittingNow()).sitting_id;
  for(let i=0;i<50 && (await chatFix()).sockets<1;i++)await new Promise(r=>setTimeout(r,100));
  assert.equal((await chatFix()).sockets,1,'the page never opened the Hub chat socket');
  // Fullscreen first: the chat box lives inside #stage, so it must work there.
  await page.click('#fullscreen');
  await page.waitForFunction(()=>document.fullscreenElement?.id==='stage',{timeout:3000});
  await page.waitForFunction(()=>document.activeElement?.id==='screen',{timeout:3000});
  for(const id of ['chatToggle','chatPanel','chatStrip','chatInput'])
    assert.equal(await page.$eval('#stage',(s,id)=>s.contains(document.getElementById(id)),id),true,id+' is outside #stage');
  // A held game button, then T: the panel opens, one neutral input releases the button, and with no
  // controller connected the game pauses (a keyboard player cannot type and play at once).
  const inputs=l=>l.retro.filter(r=>r.op==='input');
  await page.keyboard.down('ArrowRight');
  for(let i=0;i<30 && !inputs(await fixture()).some(r=>r.mask===128);i++)await new Promise(r=>setTimeout(r,50));
  assert(inputs(await fixture()).some(r=>r.mask===128),'the held button never reached the game');
  log=await fixture();const opsBefore=log.retro.length;
  await page.keyboard.press('KeyT');
  await page.waitForFunction(()=>!document.querySelector('#chatPanel').hidden && document.activeElement?.id==='chatInput',{timeout:3000});
  assert.equal(await page.$eval('#chatInput',e=>e.value),'','the T that opened the chat was typed into it');
  await page.keyboard.up('ArrowRight');
  for(let i=0;i<30;i++){log=await fixture();if(log.retro.slice(opsBefore).some(r=>r.op==='pause'))break;await new Promise(r=>setTimeout(r,100));}
  const afterOpen=log.retro.slice(opsBefore);
  assert(afterOpen.some(r=>r.op==='pause'),'opening the chat with no controller did not pause the game: '+JSON.stringify(afterOpen));
  // The page repeats the held state every 50 ms, so one repeat of the held button can land between the snapshot
  // and the key press. What must hold: a neutral input is sent, and nothing held is sent after it.
  const firstNeutral=afterOpen.findIndex(r=>r.op==='input' && r.mask===0);
  assert(firstNeutral>=0,'opening the chat sent no neutral input: '+JSON.stringify(afterOpen));
  assert(!afterOpen.slice(firstNeutral).some(r=>r.op==='input' && r.mask!==0),
    'a held button was sent after the chat opened: '+JSON.stringify(afterOpen));
  assert.equal(await page.evaluate(()=>document.fullscreenElement?.id),'stage','opening the chat left fullscreen');
  // Typing never moves the character: game keys typed into the input change no button mask.
  await page.type('#chatInput','zx as qw ');
  for(const k of ['ArrowLeft','ArrowUp','ArrowRight','ShiftLeft','KeyT'])await page.keyboard.press(k);
  assert.equal(await page.$eval('#chatPanel',e=>e.hidden),false,'T typed in the chat closed or reopened it');
  await page.$eval('#chatInput',e=>{e.value='';});
  await page.type('#chatInput','where am I?');
  await new Promise(r=>setTimeout(r,400));
  log=await fixture();
  const moved=log.retro.slice(opsBefore).filter(r=>r.op==='input' && r.mask!==0);
  assert.deepEqual(moved,[],'typing in the chat moved the character');
  assert.equal(await page.evaluate(()=>document.activeElement?.id),'chatInput');
  // Send: the stub /ws receives the contract frame; a guest is on a page-local session.
  await page.keyboard.press('Enter');
  let chatLog;
  for(let i=0;i<50;i++){chatLog=await chatFix();if(chatLog.frames.some(f=>f.text_input))break;await new Promise(r=>setTimeout(r,100));}
  const sent=chatLog.frames.find(f=>f.text_input);
  assert.deepEqual(sent,{text_input:'where am I?',mode:'orion',disable_tts:true,session_id:'retro-guest-'+sittingId,
    surface_context:{surface:'retro',input_modality:'typed',retro_sitting_id:sittingId}},JSON.stringify(sent));
  const hellos=chatLog.frames.filter(f=>f.type==='session_hello');
  assert(hellos.length>=1 && hellos.at(-1).session_id==='retro-guest-'+sittingId,'no session_hello for the guest session: '+JSON.stringify(chatLog.frames));
  assert(chatLog.frames.indexOf(hellos[0])<chatLog.frames.indexOf(sent),'the hello must come before the message');
  await page.waitForFunction(()=>document.querySelector('#chatStripText').textContent==='Reply to: where am I?' && !document.querySelector('#chatStrip').hidden,{timeout:3000});
  const panelText=await page.$eval('#chatLog',e=>e.textContent);
  assert(panelText.includes('where am I?') && panelText.includes('Reply to: where am I?'),panelText);
  assert(!panelText.includes('draft (must not show)'),'a non-final frame was shown as the reply');
  assert.equal(await page.$('#chatStrip .thumbs'),null,'a reply is not a line: no thumbs');
  // A dropped chat socket reconnects without taking focus from the typing player.
  await fetch(url+'/__fixture/chat_drop',{method:'POST'});
  await new Promise(r=>setTimeout(r,300));
  for(let i=0;i<40 && (await chatFix()).sockets<1;i++)await new Promise(r=>setTimeout(r,100));
  assert.equal((await chatFix()).sockets,1,'the chat socket did not reconnect');
  await new Promise(r=>setTimeout(r,300));
  assert.equal(await page.evaluate(()=>document.activeElement?.id),'chatInput','a reconnect stole focus');
  // A line push: the strip shows it with thumbs; a thumb POSTs and shows the stored state.
  await fetch(url+'/__fixture/push',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(
    {kind:'orion_outreach',llm_response:'That jump took three tries.',source:'retro_commentary',retro_sitting_id:sittingId,line_id:'line-1'})});
  await page.waitForFunction(()=>document.querySelector('#chatStripText').textContent==='That jump took three tries.' &&
    document.querySelector('#chatStrip .thumbs[data-line="line-1"]'),{timeout:3000});
  assert(await page.$('#chatLog .thumbs[data-line="line-1"]'),'the line is missing from the panel');
  await page.click('#chatStrip .thumbs button[data-thumb="up"]');
  for(let i=0;i<30;i++){chatLog=await chatFix();if(chatLog.thumbs.length)break;await new Promise(r=>setTimeout(r,100));}
  assert.deepEqual(chatLog.thumbs,[{line_id:'line-1',body:{thumb:'up'}}],JSON.stringify(chatLog.thumbs));
  await page.waitForFunction(()=>[...document.querySelectorAll('.thumbs[data-line="line-1"]')].every(b=>
    b.querySelector('button[data-thumb="up"]').getAttribute('aria-pressed')==='true' && b.querySelector('.thumbState').textContent.startsWith('Saved: thumb up')),{timeout:3000});
  // Orion's other messages are to Juniper: a guest at the game never sees them (review finding, patch 3).
  await fetch(url+'/__fixture/push',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(
    {kind:'orion_outreach',llm_response:'A private note for Juniper.',source:'endogenous_outreach'})});
  await new Promise(r=>setTimeout(r,600));
  assert(!(await page.$eval('#chatLog',e=>e.textContent)).includes('A private note for Juniper.'),'a guest saw outreach meant for Juniper');
  // A flip of "Who's playing?" mid-sitting changes only later messages.
  const flip=await page.evaluate(async id=>(await fetch('/api/retro/sitting/player',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({player:'juniper',sitting_id:id})})).status,sittingId);
  assert.equal(flip,200,'flip refused');
  await new Promise(r=>setTimeout(r,5600));   // one sitting poll
  await page.type('#chatInput','it is me');await page.keyboard.press('Enter');
  for(let i=0;i<50;i++){chatLog=await chatFix();if(chatLog.frames.filter(f=>f.text_input).length>=2)break;await new Promise(r=>setTimeout(r,100));}
  const [firstMsg,secondMsg]=chatLog.frames.filter(f=>f.text_input);
  assert.equal(firstMsg.session_id,'retro-guest-'+sittingId,'the earlier message was rewritten');
  assert.equal(secondMsg.session_id,'juniper-main-thread','Juniper stated: the message must use her main chat thread');
  // Juniper stated: any other outreach goes to the panel only, never the strip.
  await fetch(url+'/__fixture/push',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(
    {kind:'orion_outreach',llm_response:'A news note from Orion.',source:'endogenous_outreach'})});
  await page.waitForFunction(()=>document.querySelector('#chatLog').textContent.includes('A news note from Orion.'),{timeout:3000});
  assert(!(await page.$eval('#chatStripText',e=>e.textContent)).includes('A news note from Orion.'),'a non-line outreach reached the strip');
  assert.equal(await page.evaluate(()=>localStorage.getItem('orion_sid')),'juniper-main-thread');
  assert(chatLog.frames.some(f=>f.type==='session_hello' && f.session_id==='juniper-main-thread'),'no hello for Juniper\'s thread');
  // Close: the game resumes (we paused it) and gets the keyboard back, still in fullscreen.
  log=await fixture();const opsBeforeClose=log.retro.length;
  await page.click('#chatClose');
  await page.waitForFunction(()=>document.querySelector('#chatPanel').hidden,{timeout:2000});
  for(let i=0;i<50;i++){log=await fixture();if(log.retro.slice(opsBeforeClose).some(r=>r.op==='control'))break;await new Promise(r=>setTimeout(r,100));}
  assert(log.retro.slice(opsBeforeClose).some(r=>r.op==='resume'),'closing the chat did not resume the game');
  assert(log.retro.slice(opsBeforeClose).some(r=>r.op==='control'),'closing the chat did not take the controller back');
  await page.waitForFunction(()=>document.activeElement?.id==='screen',{timeout:3000});
  assert.equal(await page.evaluate(()=>document.fullscreenElement?.id),'stage');
  // The small button opens it too.
  await page.click('#chatToggle');
  await page.waitForFunction(()=>!document.querySelector('#chatPanel').hidden && document.activeElement?.id==='chatInput',{timeout:3000});
  await page.screenshot({path:out+'/chat-box.png'});
  await page.click('#chatToggle');
  await page.waitForFunction(()=>document.querySelector('#chatPanel').hidden,{timeout:2000});
  await page.evaluate(()=>document.exitFullscreen());
  assert.deepEqual(errors,[]);
  fs.writeFileSync(out+'/play-panel.json',JSON.stringify({chat:chatLog,atlas,goal,retro:log.retro.map(r=>r.op+(r.channel?':'+r.channel:'')),human:log.human},null,2));
}finally{await browser.close();}

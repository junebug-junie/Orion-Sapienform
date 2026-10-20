/* Same-origin operator client. Native frame IDs and sample counts travel together. */
(() => {
  'use strict';
  const el = id => document.getElementById(id), screen = el('screen'), ctx = screen.getContext('2d');
  let sid, generation, connection, lease, input, media, seq = 0, timer;
  const outstanding = new Map();
  let audio, gain, audioEnd = 0, lastSample = null, decoding = false, pendingPacket = null, transportEpoch = 0;
  const keys = new Set(), sources = new Set();
  const bits = {ArrowUp:16, ArrowDown:32, ArrowLeft:64, ArrowRight:128, KeyZ:256, KeyX:1, Enter:8, ShiftLeft:4, ShiftRight:4};
  const report = text => { el('status').textContent = text; };
  // wanted: the player chose to play, so a lost transport heals itself instead of
  // asking for Connect / resume. intent changes on every New/Pause/Stop/connect, so
  // an older connect or pending heal can never override a newer choice.
  let wanted = false, away = false, connecting = 0, intent = 0, healPending = false, heals = 0, healWindow = 0;
  let scriptedMode = false, cognitivePort = null, planning = false, plannedAttempt = null;
  function heal() {
    if (!wanted || away || !sid || connecting || lease || healPending || document.hidden || !document.hasFocus()) return;
    const now = performance.now(); if (now-healWindow > 10000) { heals = 0; healWindow = now; }
    if (++heals > 3) { wanted = false; report('Could not get the controller back. Click the game or press New game.'); return; }
    const mine = intent; healPending = true;
    setTimeout(() => { healPending = false;
      if (mine === intent && wanted && !away && !lease && !connecting && !document.hidden) connect().catch(err => report(err.message)); }, 300);
  }
  // Transport loss, a lapsed lease and an overload pause heal (bounded by heal()'s
  // 3-per-10 s budget); an explicit Pause from any window or the CLI is respected.
  // session_paused:<why> comes from the worker; only an explicit pause is honored.
  const SERVER_PAUSE = new Set(['control_revoked', 'session_paused:pause', 'session_not_running', 'stopped', 'session_stopped']);
  function lost(reason) {
    if(scriptedMode){report('Scripted controller handed off. Use Take this controller to return.');return;}
    if (SERVER_PAUSE.has(reason)) { wanted = false; report('Paused. Click the game to keep playing.'); return; }
    report('Reconnecting…'); heal();
  }
  async function api(path, data) {
    const r = await fetch('/api/retro/'+path, {method:data === undefined?'GET':'POST', headers:{'Content-Type':'application/json'}, body:data === undefined?undefined:JSON.stringify(data)});
    const value = await r.json(); if (!r.ok) throw Error(typeof value.detail==='string'?value.detail:'Request failed'); return value;
  }
  function flushAudio() { for (const s of sources) { try { s.stop(); } catch (_) {} } sources.clear(); audioEnd=0; lastSample=null; }
  async function sound() {
    if (!audio) { audio = new AudioContext({latencyHint:'interactive'}); gain=audio.createGain(); gain.connect(audio.destination); }
    gain.gain.value=el('mute').checked?0:1; await audio.resume();
  }
  function closeConnections() {
    scriptedMode=false;cognitivePort=null;
    transportEpoch++; clearInterval(timer); keys.clear(); lease=null; outstanding.clear();
    if (input) { input.onclose=null; input.close(); } if (media) { media.onclose=null; media.close(); }
    input=null;media=null; pendingPacket=null; flushAudio();
  }
  function neutral() { closeConnections(); report('Paused. Click the game to keep playing.'); }
  function mask() {
    let m=0; for(const code of keys) m |= bits[code] || 0;
    let pads=[];try { pads=navigator.getGamepads?.() || []; } catch (_) { /* Keyboard remains available over HTTP. */ }
    const pad = pads[Number(el('port').value)] || pads[0];
    if (pad && pad.mapping==='standard' && document.activeElement===screen) {
      const b=pad.buttons;
      if(b[0]?.pressed)m|=256;if(b[1]?.pressed)m|=1;if(b[8]?.pressed)m|=4;if(b[9]?.pressed)m|=8;
      if(b[12]?.pressed||pad.axes[1]<-.5)m|=16;if(b[13]?.pressed||pad.axes[1]>.5)m|=32;
      if(b[14]?.pressed||pad.axes[0]<-.5)m|=64;if(b[15]?.pressed||pad.axes[0]>.5)m|=128;
    }
    if((m&48)===48)m&=~48;if((m&192)===192)m&=~192;return m;
  }
  function send() {
    if(!lease || input?.readyState!==WebSocket.OPEN)return;
    if(outstanding.size>=24 || input.bufferedAmount>4096 ||
        (outstanding.size && performance.now()-outstanding.values().next().value>1000)){
      neutral();lost('stall');return;
    }
    outstanding.set(seq+1,performance.now());input.send(JSON.stringify({version:1,session_id:sid,generation,...lease,seq:++seq,mask:mask()}));
  }
  async function grant(action='acquire') {
    const epoch=transportEpoch;
    const result=await api(`sessions/${sid}/control`,{connection_id:connection,port:Number(el('port').value),action});
    if(epoch!==transportEpoch || document.hidden)return;
    lease=result;
    scriptedMode=false;cognitivePort=null;
    seq=0; outstanding.clear();clearInterval(timer);timer=setInterval(send,50);send();screen.focus();
  }
  const wsURL = path => `${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/retro/${path}`;
  async function connect() {
    scriptedMode=false;cognitivePort=null;
    const mine = ++intent; connecting = mine; away = false;
    try { await connectNow(); if (mine === intent) wanted = true; }
    finally { if (connecting === mine) connecting = 0; }
  }
  async function connectNow() {
    // An open input socket can be resumed directly. Replacing it races its old
    // disconnect against the resume and can interrupt another controller.
    if(input?.readyState===WebSocket.OPEN && connection){
      const epoch=transportEpoch;await sound();
      if(epoch!==transportEpoch || document.hidden)return;
      const status=await api(`sessions/${sid}`);
      if(epoch!==transportEpoch || document.hidden)return;
      if(status.state!=='running'){lease=null;outstanding.clear();clearInterval(timer);}
      await api(`sessions/${sid}/resume`,{});
      if(epoch!==transportEpoch || document.hidden)return;
      if(!lease)await grant();else screen.focus();
      report('Playing.');return;
    }
    closeConnections(); const epoch=transportEpoch; await sound();
    if(epoch!==transportEpoch || document.hidden)return;
    if(!sid) { const state=await api('health/ready');sid=state.session?.session_id; }
    if(epoch!==transportEpoch || document.hidden)return;
    if(!sid)throw Error('Start a new game first.');
    await api(`sessions/${sid}/resume`,{});
    if(epoch!==transportEpoch || document.hidden)return;
    input=new WebSocket(wsURL(`sessions/${sid}/input`));
    input.onclose=()=>{if(epoch===transportEpoch){neutral();lost('closed');}};
    input.onmessage=async e=>{
      if(epoch!==transportEpoch)return;
      const msg=JSON.parse(e.data);
      if(msg.connection_id) { connection=msg.connection_id;generation=msg.generation;
        try { await grant(); } catch(err) {report(err.message==='port_owned'?'Another window has this player. Open More to pick Player 2 or take this controller.':err.message);}
      } else if(msg.applied){
        if(!lease || msg.applied.epoch!==lease.epoch || msg.applied.port!==lease.port)return;
        outstanding.delete(msg.applied.seq);screen.dataset.appliedFrame=msg.applied.frame;
      } else {if(msg.epoch!==undefined && (!lease || msg.epoch!==lease.epoch || msg.port!==lease.port))return;
        // A newer input replaced this one before its frame; the lease is still live.
        if(msg.rejected==='superseded' || msg.rejected==='expired_in_queue'){outstanding.delete(msg.seq);return;}
        lease=null;outstanding.clear();clearInterval(timer);lost(msg.rejected||msg.unknown);}

    };
    media=new WebSocket(wsURL(`sessions/${sid}/media`)); media.binaryType='arraybuffer';
    media.onclose=()=>{if(epoch===transportEpoch){neutral();lost('closed');}};
    media.onmessage=e=>{pendingPacket={data:e.data,epoch};void drain();};
    report('Playing. Press Enter to start.');
  }
  async function drain() {
    if(decoding)return;decoding=true;
    try {while(pendingPacket){const packet=pendingPacket;pendingPacket=null;await draw(packet);}}
    catch(err){report('Media error: '+err.message);neutral();}
    finally{decoding=false;}
  }
  async function draw({data,epoch}) {
    const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate'));
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    if(epoch!==transportEpoch)return;
    const view=new DataView(bytes.buffer), n=view.getUint32(0), h=JSON.parse(new TextDecoder().decode(bytes.subarray(4,4+n)));
    if(h.version!==1 || h.width>512 || h.height>512 || h.audio_frames>16384)throw Error('Invalid packet');
    let offset=4+n;const image=ctx.createImageData(h.width,h.height), out=image.data;
    for(let i=0;i<h.width*h.height;i++){
      const v=h.format===1?view.getUint32(offset+i*4,true):view.getUint16(offset+i*2,true);
      if(h.format===1){out[i*4]=(v>>16)&255;out[i*4+1]=(v>>8)&255;out[i*4+2]=v&255;}
      else {const six=h.format===2;out[i*4]=((v>>(six?11:10))&31)*255/31;out[i*4+1]=((v>>5)&(six?63:31))*255/(six?63:31);out[i*4+2]=(v&31)*255/31;}
      out[i*4+3]=255;
    }
    offset+=h.video_bytes;
    if(audio && h.audio_frames){
      if(lastSample!==null && lastSample!==h.sample_end-h.audio_frames)flushAudio();
      if(audioEnd<audio.currentTime || audioEnd>audio.currentTime+.12)flushAudio();
      const at=Math.max(audio.currentTime+.025,audioEnd), buffer=audio.createBuffer(2,h.audio_frames,h.sample_rate);
      for(let c=0;c<2;c++){const channel=buffer.getChannelData(c);for(let i=0;i<h.audio_frames;i++)channel[i]=view.getInt16(offset+i*4+c*2,true)/32768;}
      const source=audio.createBufferSource();source.buffer=buffer;source.connect(gain);sources.add(source);source.onended=()=>sources.delete(source);source.start(at);
      audioEnd=at+buffer.duration;lastSample=h.sample_end;
      screen.dataset.audioFrames=String((Number(screen.dataset.audioFrames)||0)+h.audio_frames);
      // Present the associated frame when this packet's PCM starts, with bounded pending work.
      await new Promise(resolve=>setTimeout(resolve,Math.max(0,(at-audio.currentTime)*1000)));
      if(epoch!==transportEpoch)return;
    }
    if(screen.width!==h.width || screen.height!==h.height){screen.width=h.width;screen.height=h.height;}
    ctx.putImageData(image,0,0);screen.dataset.frame=h.frame;
  }
  async function load(){const games=await api('games');el('game').replaceChildren(...games.map(g=>{const o=document.createElement('option');o.value=g.game_id;o.textContent=g.title;o.dataset.ports=JSON.stringify(g.ports);o.dataset.save=String(g.save_enabled===true);return o;}));el('game').value=games.some(g=>g.game_id==='spacegulls')?'spacegulls':games[0]?.game_id;const state=await api('health/ready');sid=state.session?.session_id;
    el('player').hidden=false;
    if(state.session && ['running','paused','starting'].includes(state.session.state)){el('game').value=state.session.game_id;report('Game '+state.session.state+'. Click the game to play.');}
    else if(state.session?.reason==='service_restart'){el('game').value=state.session.game_id;report('The game service restarted. Choose a save and press Load to continue.');}
    else report('Choose a game and press New game.');
    await listSaves();}
  // Save states live on the server, keyed to this game's ROM and core. Saves from
  // another ROM/core build are hidden here and refused by the server anyway.
  const played = f => { const t=Math.floor(f/60); return `${Math.floor(t/3600)}:${String(Math.floor(t/60)%60).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`; };
  async function listSaves(){
    const game=el('game').value, enabled=el('game').selectedOptions[0]?.dataset.save==='true';
    el('saveBar').hidden=!enabled; if(!enabled)return;
    const rows=(await api(`games/${game}/saves`)).filter(s=>s.compatible);
    if(game!==el('game').value)return;
    el('saves').replaceChildren(...rows.map(s=>{const o=document.createElement('option');o.value=s.save_id;
      o.textContent=`${new Date(s.created_utc).toLocaleString()} · played ${played(s.played_frames)} · frame ${s.frame}`;return o;}));
    if(!rows.length){const o=document.createElement('option');o.value='';o.textContent='No saves yet';el('saves').append(o);}
  }
  el('game').onchange=()=>{void listSaves().catch(err=>report(err.message));};
  const action = (id, fn) => el(id).onclick=async()=>{el(id).disabled=true;try{await fn();}catch(err){report(err.message);}finally{el(id).disabled=false;}};
  async function newGame(save_id){++intent;wanted=false;
    // One game exists at a time: New game ends the current one instead of failing with session_active.
    if(sid){closeConnections();try{await api(`sessions/${sid}/stop`,{});}catch(err){if(err.message!=='unknown_session')throw err;}sid=null;}
    const epoch=transportEpoch;await sound();if(epoch!==transportEpoch || document.hidden)return false;
    const body={game_id:el('game').value,idempotency_key:Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,"0")).join("")};
    if(save_id)body.save_id=save_id;
    const s=await api('sessions',body);sid=s.session_id;if(epoch!==transportEpoch || document.hidden)return false;
    await connect();return true;}
  action('new',()=>newGame());
  action('save',async()=>{
    if(!sid)throw Error('Start or load a game first.');
    const r=await api(`sessions/${sid}/save`,{});
    await listSaves();
    report(`Saved · played ${played(r.saved.played_frames)} · frame ${r.saved.frame}.`);
    if(lease)screen.focus();
  });
  action('load',async()=>{
    const save_id=el('saves').value;if(!save_id)throw Error('No save to load.');
    let current=null;
    if(sid){try{current=await api(`sessions/${sid}`);}catch(_){current=null;}}
    if(current && current.game_id===el('game').value && ['running','paused'].includes(current.state)){
      // Restore in place; the worker swaps the whole game state between two frames.
      await api(`sessions/${sid}/load`,{save_id});
      if(lease && current.state==='running'){screen.focus();report('Save loaded. Playing.');}
      else report('Save loaded (paused; the picture updates when play resumes). Click the game to keep playing.');
    } else if(await newGame(save_id)) report('Save loaded. Playing.');
  });
  action('join',connect);
  action('pause',async()=>{++intent;wanted=false;try{if(sid)await api(`sessions/${sid}/pause`,{});}finally{neutral();}report('Paused. Click the game to keep playing.');});
  action('stop',async()=>{++intent;wanted=false;try{if(sid)await api(`sessions/${sid}/stop`,{});}finally{neutral();}sid=null;report('Game stopped.');});
  action('takeover',async()=>{
    if(input?.readyState!==WebSocket.OPEN){await connect();return;}
    const epoch=transportEpoch;await api(`sessions/${sid}/resume`,{});
    if(epoch!==transportEpoch || document.hidden)return;
    await grant('takeover');wanted=true;report('Playing.');
  });
  async function handoff(cognition=false){
    if(!lease || input?.readyState!==WebSocket.OPEN)throw Error('Connect to this player before handing it to the script.');
    const port=lease.port, expected_epoch=lease.epoch, epoch=transportEpoch;
    ++intent;wanted=false;scriptedMode=true;lease=null;keys.clear();outstanding.clear();clearInterval(timer);
    try{
      await api(`sessions/${sid}/scripted`,{connection_id:connection,generation,port,expected_epoch,action:'start',duration_seconds:60,cognition});
    }catch(err){scriptedMode=false;cognitivePort=null;throw Error(`Handoff failed: ${err.message}. Use Take this controller to return.`);}
    if(epoch!==transportEpoch)return;
    cognitivePort=cognition?port:null;plannedAttempt=null;
    report(cognition?'Orion goal experiment — 60 seconds. Take this controller to interrupt.':'Scripted right/jump baseline — 60 seconds. Take this controller to interrupt.');
    await showPolicy();
  }
  action('scripted',()=>handoff());
  action('cognitive',()=>handoff(true));
  async function showPolicy(){
    if(!sid)return;
    const current=sid, state=await api(`sessions/${sid}`);
    if(current!==sid)return;
    if(state.state==='paused' && !(state.scripted||[]).length){scriptedMode=false;cognitivePort=null;}
    el('policy').textContent=(state.scripted||[]).length
      ? state.scripted.map(p=>`Player ${p.port+1}: ${p.policy_id} (${p.cognition?'goal experiment':'scripted'}, ${p.duration_seconds}s limit)`).join(' · ')
      : 'No scripted controller active.';
    const policy=(state.scripted||[]).find(p=>p.port===cognitivePort && p.cognition);
    if(!policy)return;
    const facts=policy.cognition, attempt=facts.attempt;
    el('goal').textContent=`${facts.goal?.goal||'wait'}: ${facts.goal?.rationale||'Waiting for an observed attempt.'} Outcome: ${attempt?.outcome||'unknown'}`;
    if(!attempt || planning || plannedAttempt===attempt.attempt_id)return;
    planning=true;plannedAttempt=attempt.attempt_id;
    // Planner requests never block input/media or the native frame loop.
    const epoch=transportEpoch, currentConnection=connection;
    void api(`sessions/${current}/plan`,{connection_id:connection,generation,
      port:policy.port,expected_epoch:policy.epoch}).then(result=>{
        if(epoch===transportEpoch && connection===currentConnection && cognitivePort!==null)
          el('goal').textContent=`${result.goal}: ${result.rationale}`;
      }).catch(err=>{
        if(epoch===transportEpoch)el('goal').textContent=`Planner unavailable: ${err.message}. Motor returns to wait; takeover remains available.`;
      }).finally(()=>{planning=false;});
  }
  // Read-only ownership display also tells a second window which port is scripted.
  let pollingPolicy=false;
  setInterval(async()=>{if(document.hidden || pollingPolicy)return;pollingPolicy=true;
    try{await showPolicy();}catch(_){el('policy').textContent='Controller status unavailable.';}
    finally{pollingPolicy=false;}},1000);
  action('start',async()=>{screen.focus();keys.add('Enter');send();setTimeout(()=>keys.delete('Enter'),150);});
  el('mute').onchange=()=>{if(gain)gain.gain.value=el('mute').checked?0:1;};
  el('port').onchange=neutral;
  // While the game has the keyboard, keys that scroll a page do nothing else. Before
  // this, unmapped PageUp/PageDown/Home/End/Space fell through to the browser and
  // scrolled the Hub page (the Retro frame is taller than the window) out from under
  // the game. In fullscreen, mapped keys reach the game even if focus slipped off it.
  const stage=el('stage'), SCROLL_KEYS=new Set(['PageUp','PageDown','Home','End','Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight']);
  const field=t=>t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement;
  document.addEventListener('keydown',e=>{
    if(document.activeElement!==screen && (document.fullscreenElement!==stage || field(e.target)))return;
    if(bits[e.code] || SCROLL_KEYS.has(e.code))e.preventDefault();
    if(bits[e.code] && !keys.has(e.code)){keys.add(e.code);send();}
  },{capture:true});
  action('fullscreen',async()=>{
    if(document.fullscreenElement){await document.exitFullscreen();return;}
    if(!stage.requestFullscreen || !document.fullscreenEnabled)throw Error('Fullscreen is blocked here. Open /retro directly.');
    await stage.requestFullscreen({navigationUI:'hide'});
  });
  el('exitFullscreen').onclick=()=>{if(document.fullscreenElement)void document.exitFullscreen().catch(()=>{});};
  document.addEventListener('fullscreenchange',()=>{
    el('fullscreen').textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';screen.focus();});
  window.addEventListener('keyup',e=>{keys.delete(e.code);send();});
  window.addEventListener('blur',()=>{away=true;neutral();});window.addEventListener('pagehide',neutral);window.addEventListener('gamepaddisconnected',neutral);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){away=true;neutral();}else{away=false;heal();}});
  window.addEventListener('focus',()=>{away=false;heal();});
  screen.addEventListener('pointerdown',()=>{away=false;if(scriptedMode)return;if(sid)wanted=true;heals=0;heal();});
  void load().catch(err=>{el('player').hidden=false;report(err.message);});
})();

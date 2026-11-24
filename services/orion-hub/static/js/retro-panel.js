/* Same-origin operator client. Native frame IDs and sample counts travel together. */
(() => {
  'use strict';
  const el = id => document.getElementById(id), screen = el('screen'), ctx = screen.getContext('2d');
  let sid, generation, connection, lease, input, media, seq = 0, timer, acquiring=null;
  const outstanding = new Map();
  let audio, gain, audioEnd = 0, lastSample = null, decoding = false, pendingPacket = null, transportEpoch = 0;
  let mediaRetry, mediaHeals=0, mediaWindow=0, lastPictureAt=0, pictureReady=false;
  let conditionalResume=false; // older native services ignore resume bodies: fail closed
  let mediaFlowControl=false; // negotiate with the gateway; old pages keep legacy media
  let activeSystem='nes'; // follows the running session, never an unstarted dropdown choice
  const keys = new Set(), sources = new Set();
  // Game company patch 3: the chat box. While its input is open, nothing puts focus back on the
  // game canvas: a reconnect, a grant, a save or a fullscreen change would steal the player's typing.
  let chatOpen = false;
  function focusGame(){ if(!chatOpen)screen.focus(); }
  // N64 (Mupen64Plus-Next standard map): X = A, A = B, C = Z, Q/W = L/R. Arrows push the
  // control stick and I/J/K/L the C stick instead of the D-pad. Select is not an N64 button.
  // Bit 15 marks the C-stick keys only: no console accepts it, so mask() always removes it.
  const bits = {ArrowUp:16, ArrowDown:32, ArrowLeft:64, ArrowRight:128, KeyZ:256, KeyX:1, KeyA:2, KeyS:512, KeyQ:1024, KeyW:2048, Enter:8, ShiftLeft:4, ShiftRight:4, KeyC:4096, KeyI:32768, KeyJ:32768, KeyK:32768, KeyL:32768},
    SYSTEM_BUTTONS = {nes:509, snes:4095, n64:16379},
    STICK = {ArrowLeft:[0,-1], ArrowRight:[0,1], ArrowUp:[1,-1], ArrowDown:[1,1], KeyJ:[2,-1], KeyL:[2,1], KeyI:[3,-1], KeyK:[3,1]};
  const SYSTEM_LABEL = {snes:' (SNES)', n64:' (N64)'};
  // Retro-Bit Tribute64 (USB receiver, Xbox mode) on N64: its four yellow C buttons arrive as
  // plain buttons, standard index -> [axis, direction]: Y = C-left, left-stick click = C-right,
  // right-stick click = C-up, X = C-down. Its B is the right face button, Z the left trigger
  // (L2, already N64 Z). Read from the pad's
  // raw input on carbon-x1, 2026-11-24. It names itself an Xbox 360 pad, like an 8BitDo in X
  // mode, so the player turns this on in More; it is kept per browser.
  const TRIBUTE_C = {3:[2,-1], 10:[2,1], 11:[3,-1], 2:[3,1]};
  let tribute64=false;try{tribute64=localStorage.getItem('retro_tribute64')==='1';}catch(_){}
  // A key is a game key only when this console has its button (or, on N64, its stick).
  const gameKey = code => !!((bits[code]||0)&(SYSTEM_BUTTONS[activeSystem] ?? 509) || activeSystem==='n64'&&STICK[code]);
  const titleKey = g => (g.title+(SYSTEM_LABEL[g.system]||'')).replace(/^the\s+/i,'').toLowerCase();
  // #status sits outside the fullscreen stage, so mirror each message into a toast
  // shown only in fullscreen; otherwise F7/F10 results would be invisible there.
  let toastTimer=null;
  const report = text => { el('status').textContent = text; el('toast').textContent = text;
    clearTimeout(toastTimer); toastTimer=setTimeout(()=>{el('toast').textContent='';},3000); };
  // wanted: the player chose to play, so a lost transport heals itself instead of
  // asking for Connect / resume. intent changes on every New/Pause/Stop/connect, so
  // an older connect or pending heal can never override a newer choice.
  let wanted = false, away = false, connecting = 0, intent = 0, healPending = false, heals = 0, healWindow = 0;
  let scriptedMode = false, cognitivePort = null, planning = false, plannedAttempt = null;
  // "Let Orion play": hand-overs of ORION_PLAY_SEC (Retro's limit), renewed when one ends, until the
  // player takes the game back (click, a game key, a pad button, Take this controller) or the tab hides.
  const ORION_PLAY_SEC = 1800;
  let orionPlay = false, orionRenewing = false, orionGrant = null;   // orionGrant: {started_frame, duration_seconds}
  function setOrion(on){
    orionPlay=on;
    const b=el('orionplay'); if(b){b.setAttribute('aria-pressed',String(on));b.textContent=on?'Orion is playing':'Let Orion play';}
  }
  // The grant this page handed off ({port, epoch}); epoch latches on the first status that shows it.
  let handoffGrant = null;
  // Zelda (planner v2): one plan per Retro decision; a heat shed waits out retry_after_sec.
  let plannedDecision = null, planRetryAt = 0, watching = false;
  function heal() {
    if (!wanted || away || !sid || connecting || acquiring || lease || healPending || document.hidden || !document.hasFocus()) return;
    const now = performance.now(); if (now-healWindow > 10000) { heals = 0; healWindow = now; }
    if (++heals > 3) { wanted = false; report('Could not get the controller back. Click the game or press New game.'); return; }
    const mine = intent, session=sid, epoch=transportEpoch; healPending = true;
    setTimeout(() => { healPending = false;
      if (mine === intent && session===sid && epoch===transportEpoch && wanted && !away && !lease && !connecting && !document.hidden && document.hasFocus())
        connect(true).catch(err => {report(err.message);heal();}); }, 300);
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
  async function api(path, data, signal) {
    const r = await fetch('/api/retro/'+path, {method:data === undefined?'GET':'POST', headers:{'Content-Type':'application/json'}, body:data === undefined?undefined:JSON.stringify(data),signal});
    const value = await r.json();
    if (!r.ok) { const e=Error(typeof value.detail==='string'?value.detail:'Request failed'); e.status=r.status; e.body=value; throw e; }
    return value;
  }
  function flushAudio() { for (const s of sources) { try { s.stop(); } catch (_) {} } sources.clear(); audioEnd=0; lastSample=null; }
  async function sound() {
    if (!audio) { audio = new AudioContext({latencyHint:'interactive'}); gain=audio.createGain(); gain.connect(audio.destination); }
    gain.gain.value=el('mute').checked?0:1; await audio.resume();
  }
  function closeConnections() {
    scriptedMode=false;cognitivePort=null;handoffGrant=null;watching=false;
    transportEpoch++; acquiring=null;clearInterval(timer); clearTimeout(mediaRetry); keys.clear(); lease=null; outstanding.clear();
    if (input) { input.onclose=null; input.close(); } if (media) { media.onclose=null; media.close(); }
    input=null;media=null; pendingPacket=null; flushAudio(); resetChord(chord);
  }
  function neutral() { setOrion(false); closeConnections(); report('Paused. Click the game to keep playing.'); }
  function openMedia(epoch) {
    const session=sid, flow=mediaFlowControl;
    const socket=new WebSocket(wsURL(`sessions/${sid}/media`)+(flow?'?flow=ack-v1':''));
    let received=0;
    media=socket;socket.binaryType='arraybuffer';
    socket.onclose=()=>{
      if(epoch!==transportEpoch || session!==sid || media!==socket)return;
      media=null;pictureReady=false;pendingPacket=null;flushAudio();
      if(lease)send();  // release held buttons while the picture is unavailable
      const now=performance.now();
      if(now-mediaWindow>10000){mediaHeals=0;mediaWindow=now;}
      if(++mediaHeals>3){transportLost('media_stalled');return;}
      mediaRetry=setTimeout(()=>{
        if(epoch===transportEpoch && session===sid && !document.hidden && (input || watching))openMedia(epoch);
      },300);
    };
    socket.onmessage=e=>{if(epoch===transportEpoch && session===sid && media===socket){pendingPacket={data:e.data,epoch,socket,flow,sequence:++received};void drain();}};
  }
  function transportLost(reason) {
    const handoff=scriptedMode;
    neutral();
    if(handoff){wanted=false;report('Orion stopped: connection lost. Use Take this controller to return.');return;}
    lost(reason);
  }
  // A connected socket is not proof of a usable picture. Refresh only neutral input
  // after 1 s without a presented frame; close the sponsor/controller after 3 s.
  setInterval(()=>{
    if(input && (lease || scriptedMode) && performance.now()-lastPictureAt>3000)transportLost('media_stalled');
  },100);
  function mask() {
    let m=0; for(const code of keys) if(!(activeSystem==='n64' && STICK[code])) m |= bits[code] || 0;  // N64 arrows are the stick
    const pad = gamepad();
    if (pad && (document.activeElement===screen || document.fullscreenElement===stage || chatOpen))
      m |= selectChord(chord, pad, activeSystem==='n64'&&tribute64?tributeMask(pad):padMask(pad, activeSystem==='n64'),
        performance.now(), code=>runHotkey(code));
    else resetChord(chord);
    m&=SYSTEM_BUTTONS[activeSystem] ?? 509; // each authority refuses buttons its console lacks
    if((m&48)===48)m&=~48;if((m&192)===192)m&=~192;return m;
  }
  // N64 sticks: [left x, left y, right x, right y] in libretro's range (up is negative, as in
  // the Gamepad API). Keys push full deflection; a pad stick past its own centre wins.
  function sticks() {
    const v=[0,0,0,0];
    for(const code of keys){const s=STICK[code];if(s)v[s[0]]+=s[1];}
    const pad=gamepad();
    if(pad && (document.activeElement===screen || document.fullscreenElement===stage || chatOpen)){
      const a=pad.axes||[];for(let i=0;i<4;i++)if(Math.abs(a[i]||0)>.08)v[i]=a[i];
      if(tribute64&&activeSystem==='n64')for(const [i,[axis,dir]] of Object.entries(TRIBUTE_C))if(pad.buttons[i]?.pressed)v[axis]=dir;
    }
    return v.map(x=>Math.round(Math.max(-1,Math.min(1,x))*32767));
  }
  // Pad Select + L2 saves, Select + R2 loads the newest save (the F7 / F10 hotkeys). Select is a
  // game button (Zelda: pause and map), so the pad's Select is held back until it is released:
  // used in a chord, the game never sees it; released alone, the game gets a 150 ms tap.
  const chord={held:false, chorded:false, pulseUntil:0, waitUp:false};
  // Focus or connection lost: forget the hold, and ignore a Select still down until it is
  // released, so coming back never taps a Select nobody pressed (review 2026-10-27).
  function resetChord(st) { st.waitUp=st.waitUp || st.held; st.held=false; st.chorded=false; st.pulseUntil=0; }
  function selectChord(st, pad, m, now, fire) {
    const b=pad.buttons, sel=!!b[8]?.pressed;
    const l2=!!(b[6]?.pressed || b[6]?.value>.5), r2=!!(b[7]?.pressed || b[7]?.value>.5);
    m&=~4;
    if(st.waitUp){if(!sel)st.waitUp=false;return m;}
    if(sel){
      if(!st.held){st.held=true;st.chorded=false;}
      if((l2||r2) && !st.chorded){st.chorded=true;fire(l2?'F7':'F10');}
    } else if(st.held){st.held=false;if(!st.chorded)st.pulseUntil=now+150;}
    if(now<st.pulseUntil)m|=4;
    return m;
  }
  // Connected pads in the browser's standard mapping: this player's (Player 2 takes the second
  // pad), else the first. getGamepads() lists empty slots as null, and over plain HTTP some
  // browsers throw: the keyboard still works then.
  function gamepad() {
    let pads=[];try { pads=Array.from(navigator.getGamepads?.() || []); } catch (_) { return null; }
    const std=pads.filter(p=>p && p.connected!==false && p.mapping==='standard');
    return std[Number(el('port').value)] || std[0] || null;
  }
  // Standard mapping is positional: A right, B bottom, X top, Y left (8BitDo labels).
  // SNES uses L1/R1; L2/R2 remain save/load chords. mask() removes SNES-only bits for NES.
  function padMask(pad, n64=false) {
    const b=pad.buttons, a=pad.axes||[]; let m=0;
    if(b[1]?.pressed)m|=256;if(b[0]?.pressed)m|=1;if(b[8]?.pressed)m|=4;if(b[9]?.pressed)m|=8;
    if(b[3]?.pressed)m|=512;if(b[2]?.pressed)m|=2;if(b[4]?.pressed)m|=1024;if(b[5]?.pressed)m|=2048;
    const stick=!n64; // on N64 the left stick is analog, not the D-pad
    if(b[12]?.pressed||stick&&a[1]<-.5)m|=16;if(b[13]?.pressed||stick&&a[1]>.5)m|=32;
    if(b[14]?.pressed||stick&&a[0]<-.5)m|=64;if(b[15]?.pressed||stick&&a[0]>.5)m|=128;
    // N64 Z on L2, unless Select is held for the save/load chord.
    if(!stick && !b[8]?.pressed && (b[6]?.pressed || b[6]?.value>.5))m|=4096;
    return m;
  }
  // Tribute64 on N64: A = N64 A, B = N64 B (the core reads B from libretro Y, bit 2); its
  // C buttons leave the button mask and become the C stick in sticks().
  function tributeMask(pad) {
    let m=padMask(pad, true)&~(256|2|512);
    if(pad.buttons[1]?.pressed)m|=2;
    return m;
  }
  function padStatus() {
    let pads=[];try { pads=Array.from(navigator.getGamepads?.() || []).filter(p=>p && p.connected!==false); } catch (_) { /* no API */ }
    const pad=gamepad(), other=pads.find(p=>p.mapping!=='standard');
    el('padInfo').textContent = pad ? `Controller: ${pad.id}${tribute64?' (Tribute64 layout on for N64)':''}. Click the game, then play.`
      : other ? `Controller ${other.id} is not in standard mode. On an 8BitDo Pro 2, set the mode switch on the back to X and reconnect (a USB cable if Bluetooth still says this).`
      : 'No controller found. Connect it, then press any button.';
  }
  function send() {
    if(!lease || input?.readyState!==WebSocket.OPEN)return;
    if(outstanding.size>=24 || input.bufferedAmount>4096 ||
        (outstanding.size && performance.now()-outstanding.values().next().value>1000)){
      transportLost('stall');return;
    }
    const live=pictureReady && performance.now()-lastPictureAt<=1000;
    const currentMask=live?mask():0, axes=activeSystem==='n64'?{axes:live?sticks():[0,0,0,0]}:{};
    outstanding.set(seq+1,performance.now());input.send(JSON.stringify({version:1,session_id:sid,generation,...lease,seq:++seq,mask:currentMask,...axes}));
  }
  async function grant(action='acquire') {
    const epoch=transportEpoch, mine=intent;
    const session=sid, owner=connection, port=Number(el('port').value);
    const pending={epoch,mine,session};acquiring=pending;
    const controller=new AbortController();
    const deadline=setTimeout(()=>{
      controller.abort();
      if(acquiring===pending && epoch===transportEpoch && mine===intent){
        transportLost('acquire_timeout');
        const recoveryEpoch=transportEpoch;
        // A reused-input connect still holds `connecting` until abort unwinds.
        // Retry on the next task, after those promise finally blocks have run.
        setTimeout(()=>{if(recoveryEpoch===transportEpoch && mine===intent && session===sid)heal();},0);
      }
    },5000);
    try{
      let result;
      for(let attempt=0;attempt<2;attempt++){
        try{result=await api(`sessions/${session}/control`,{connection_id:owner,port,action},controller.signal);break;}
        catch(err){
          if(acquiring!==pending || epoch!==transportEpoch || mine!==intent || session!==sid || document.hidden)return;
          // Only recover automatic pauses, never a newer Pause or another owner.
          if(attempt || !conditionalResume || !['session_paused:disconnect','session_paused:no_controller','session_paused:deadline_overrun'].includes(err.message))throw err;
          await api(`sessions/${session}/resume`,{expected_pause_reason:err.message.slice('session_paused:'.length)},controller.signal);
          if(acquiring!==pending || epoch!==transportEpoch || mine!==intent || session!==sid || document.hidden)return;
        }
      }
      if(acquiring!==pending || epoch!==transportEpoch || mine!==intent || session!==sid || document.hidden)return;
      lease=result;
      scriptedMode=false;cognitivePort=null;handoffGrant=null;
      seq=0; outstanding.clear();clearInterval(timer);timer=setInterval(send,50);send();focusGame();
    }finally{clearTimeout(deadline);if(acquiring===pending)acquiring=null;}
  }
  const wsURL = path => `${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/retro/${path}`;
  async function connect(recovering=false) {
    if(acquiring)return; // a delayed successful grant still belongs to this intent
    scriptedMode=false;cognitivePort=null;handoffGrant=null;
    const mine = ++intent; connecting = mine; away = false;
    try { await connectNow(recovering); if (mine === intent) wanted = true; }
    finally { if (connecting === mine) connecting = 0; }
  }
  async function connectNow(recovering=false) {
    // An open input socket can be resumed directly. Replacing it races its old
    // disconnect against the resume and can interrupt another controller.
    if(input?.readyState===WebSocket.OPEN && connection){
      const epoch=transportEpoch;await sound();
      if(epoch!==transportEpoch || document.hidden)return;
      const status=await api(`sessions/${sid}`);
      if(epoch!==transportEpoch || document.hidden)return;
      if(status.state!=='running'){lease=null;outstanding.clear();clearInterval(timer);}
      if(!recovering)await api(`sessions/${sid}/resume`,{});
      if(epoch!==transportEpoch || document.hidden)return;
      if(!lease)await grant();else focusGame();
      if(epoch!==transportEpoch || document.hidden)return;
      report('Playing.');return;
    }
    closeConnections(); const epoch=transportEpoch; await sound();
    if(epoch!==transportEpoch || document.hidden)return;
    if(!sid) { const state=await api('health/ready');sid=state.session?.session_id;activeSystem=state.session?.system||'nes';conditionalResume=state.capabilities?.conditional_resume===true;mediaFlowControl=state.capabilities?.retro_media_ack_v1===true; }
    if(epoch!==transportEpoch || document.hidden)return;
    if(!sid)throw Error('Start a new game first.');
    if(!recovering)await api(`sessions/${sid}/resume`,{});
    if(epoch!==transportEpoch || document.hidden)return;
    input=new WebSocket(wsURL(`sessions/${sid}/input`));
    lastPictureAt=performance.now();pictureReady=false;mediaHeals=0;mediaWindow=lastPictureAt;
    input.onclose=()=>{if(epoch===transportEpoch)transportLost('closed');};
    input.onmessage=async e=>{
      if(epoch!==transportEpoch)return;
      const msg=JSON.parse(e.data);
      if(msg.connection_id) { connection=msg.connection_id;generation=msg.generation;
        try { await grant(); } catch(err) {
          if(epoch!==transportEpoch)return;
          neutral();
          if(err.message==='port_owned' || SERVER_PAUSE.has(err.message)){wanted=false;report(err.message==='port_owned'?'Another window has this player. Open More to pick Player 2 or take this controller.':'Paused. Click the game to keep playing.');}
          else lost(err.message);
        }
      } else if(msg.applied){
        if(!lease || msg.applied.epoch!==lease.epoch || msg.applied.port!==lease.port)return;
        outstanding.delete(msg.applied.seq);screen.dataset.appliedFrame=msg.applied.frame;
      } else {if(msg.epoch!==undefined && (!lease || msg.epoch!==lease.epoch || msg.port!==lease.port))return;
        // A newer input replaced this one before its frame; the lease is still live.
        if(msg.rejected==='superseded' || msg.rejected==='expired_in_queue'){outstanding.delete(msg.seq);return;}
        lease=null;outstanding.clear();clearInterval(timer);lost(msg.rejected||msg.unknown);}

    };
    openMedia(epoch);
    report('Playing. Press Enter to start.');
  }
  async function drain() {
    if(decoding)return;decoding=true;
    try {while(pendingPacket){
      const packet=pendingPacket;pendingPacket=null;
      try{await draw(packet);}
      catch(err){
        if(packet.epoch===transportEpoch && packet.socket===media){report('Media error: '+err.message);neutral();}
      }
    }}
    finally{decoding=false;}
  }
  async function draw({data,epoch,socket,flow,sequence}) {
    const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate'));
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    if(epoch!==transportEpoch || socket!==media)return;
    const view=new DataView(bytes.buffer), n=view.getUint32(0), h=JSON.parse(new TextDecoder().decode(bytes.subarray(4,4+n)));
    if(h.version!==1 || h.width>512 || h.height>512 || h.audio_frames>16384)throw Error('Invalid packet');
    let offset=4+n;
    // A core warming up (N64/Angrylion, first picture at ~frame 64) sends audio-only packets.
    // Drawing them would throw on a 0x0 image and close the socket, pausing the session before
    // the renderer ever produces a frame. Accept them as sound with no picture.
    const picture=h.width>0 && h.height>0;
    let image=null;
    if(picture){
      image=ctx.createImageData(h.width,h.height); const out=image.data;
      for(let i=0;i<h.width*h.height;i++){
        const v=h.format===1?view.getUint32(offset+i*4,true):view.getUint16(offset+i*2,true);
        if(h.format===1){out[i*4]=(v>>16)&255;out[i*4+1]=(v>>8)&255;out[i*4+2]=v&255;}
        else {const six=h.format===2;out[i*4]=((v>>(six?11:10))&31)*255/31;out[i*4+1]=((v>>5)&(six?63:31))*255/(six?63:31);out[i*4+2]=(v&31)*255/31;}
        out[i*4+3]=255;
      }
    }
    offset+=h.video_bytes;
    if(audio && h.audio_frames){
      if(lastSample!==null && lastSample!==h.sample_end-h.audio_frames)flushAudio();
      // Keep sound close to the picture even after a burst of delayed packets.
      if(audioEnd<audio.currentTime || audioEnd>audio.currentTime+.05)flushAudio();
      const at=Math.max(audio.currentTime+.025,audioEnd), buffer=audio.createBuffer(2,h.audio_frames,h.sample_rate);
      for(let c=0;c<2;c++){const channel=buffer.getChannelData(c);for(let i=0;i<h.audio_frames;i++)channel[i]=view.getInt16(offset+i*4+c*2,true)/32768;}
      const source=audio.createBufferSource();source.buffer=buffer;source.connect(gain);sources.add(source);source.onended=()=>sources.delete(source);source.start(at);
      audioEnd=at+buffer.duration;lastSample=h.sample_end;
      screen.dataset.audioFrames=String((Number(screen.dataset.audioFrames)||0)+h.audio_frames);
    }
    // The audio device schedules PCM independently. Waiting for its start here
    // blocks decoding and presentation credit, adding delay to every round trip.
    if(picture){
      if(screen.width!==h.width || screen.height!==h.height){screen.width=h.width;screen.height=h.height;}
      ctx.putImageData(image,0,0);screen.dataset.frame=h.frame;
      lastPictureAt=performance.now();pictureReady=true;
    }
    if(flow && socket.readyState===WebSocket.OPEN)socket.send(JSON.stringify({ack:sequence}));
  }
  async function load(){const games=(await api('games')).sort((x,y)=>titleKey(x).localeCompare(titleKey(y)));el('game').replaceChildren(...games.map(g=>{const o=document.createElement('option');o.value=g.game_id;o.textContent=g.title+(SYSTEM_LABEL[g.system]||'');o.dataset.ports=JSON.stringify(g.ports);o.dataset.save=String(g.save_enabled===true);return o;}));el('game').value=games.some(g=>g.game_id==='spacegulls')?'spacegulls':games[0]?.game_id;const state=await api('health/ready');sid=state.session?.session_id;activeSystem=state.session?.system||'nes';conditionalResume=state.capabilities?.conditional_resume===true;mediaFlowControl=state.capabilities?.retro_media_ack_v1===true;
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
    const s=await api('sessions',body);sid=s.session_id;activeSystem=s.system||'nes';if(epoch!==transportEpoch || document.hidden)return false;
    await connect();return true;}
  action('new',()=>newGame());
  action('save',async()=>{
    if(!sid)throw Error('Start or load a game first.');
    const r=await api(`sessions/${sid}/save`,{});
    await listSaves();
    report(`Saved · played ${played(r.saved.played_frames)} · frame ${r.saved.frame}.`);
    if(lease)focusGame();
  });
  let hotkeyLoad=false;  // F10 only restores in place; it never ends one game to start another
  action('load',async()=>{
    const inPlaceOnly=hotkeyLoad;hotkeyLoad=false;
    const save_id=el('saves').value;if(!save_id)throw Error('No save to load.');
    let current=null;
    if(sid){try{current=await api(`sessions/${sid}`);}catch(_){current=null;}}
    if(current && current.game_id===el('game').value && ['running','paused'].includes(current.state)){
      // Restore in place; the worker swaps the whole game state between two frames.
      await api(`sessions/${sid}/load`,{save_id});
      if(lease && current.state==='running'){focusGame();report('Save loaded. Playing.');}
      else report('Save loaded (paused; the picture updates when play resumes). Click the game to keep playing.');
    } else if(inPlaceOnly) throw Error('F10 loads into the game that is running. To start a game from a save, choose it and press Load.');
    else if(await newGame(save_id)) report('Save loaded. Playing.');
  });
  action('join',connect);
  async function pauseGame(){const session=sid;++intent;wanted=false;neutral();if(session)await api(`sessions/${session}/pause`,{});report('Paused. Click the game to keep playing.');}
  action('pause',pauseGame);
  action('stop',async()=>{const session=sid;++intent;wanted=false;neutral();sid=null;if(session)await api(`sessions/${session}/stop`,{});report('Game stopped.');});
  action('takeover',async()=>{
    setOrion(false);
    if(input?.readyState!==WebSocket.OPEN){await connect();return;}
    const epoch=transportEpoch;await api(`sessions/${sid}/resume`,{});
    if(epoch!==transportEpoch || document.hidden)return;
    await grant('takeover');wanted=true;report('Playing.');
  });
  async function handoff(cognition=false, seconds=60){
    if(!lease || input?.readyState!==WebSocket.OPEN)throw Error('Connect to this player before handing it to the script.');
    const port=lease.port, expected_epoch=lease.epoch, epoch=transportEpoch;
    ++intent;wanted=false;scriptedMode=true;handoffGrant=null;lease=null;keys.clear();outstanding.clear();clearInterval(timer);
    try{
      await api(`sessions/${sid}/scripted`,{connection_id:connection,generation,port,expected_epoch,action:'start',duration_seconds:seconds,cognition});
    }catch(err){scriptedMode=false;cognitivePort=null;throw Error(`Handoff failed: ${err.message}. Use Take this controller to return.`);}
    if(epoch!==transportEpoch)return;
    cognitivePort=cognition?port:null;handoffGrant={port,epoch:null};plannedAttempt=null;plannedDecision=null;planRetryAt=0;
    report(orionPlay?'Orion is playing. Click the game, or press any key or controller button, to take it back.'
      :cognition?'Orion goal experiment — 60 seconds. Take this controller to interrupt.':'Scripted right/jump baseline — 60 seconds. Take this controller to interrupt.');
    await showPolicy();
  }
  action('scripted',()=>{setOrion(false);return handoff();});
  action('cognitive',()=>{setOrion(false);return handoff(true);});
  // Take the controller (connect if needed), then hand it to Orion for a long turn.
  async function orionTurn(){
    if(!lease || input?.readyState!==WebSocket.OPEN){
      if(input?.readyState!==WebSocket.OPEN)await connect();   // the grant arrives on the socket's first message
      else {await api(`sessions/${sid}/resume`,{});await grant('takeover');}
      for(let i=0;i<50 && orionPlay && !(lease && input?.readyState===WebSocket.OPEN);i++)
        await new Promise(r=>setTimeout(r,100));
    }
    if(!orionPlay)return;
    await handoff(true, ORION_PLAY_SEC);
  }
  action('orionplay',async()=>{
    if(!sid)throw Error('Start or load a game first.');
    setOrion(true);
    try{await orionTurn();}catch(err){setOrion(false);throw err;}
  });
  function renewOrion(){
    if(!orionPlay || orionRenewing || document.hidden)return;
    orionRenewing=true;
    void orionTurn().catch(err=>{const was=orionPlay;setOrion(false);if(was)report(`Orion stopped: ${err.message}`);})
      .finally(()=>{orionRenewing=false;});
  }
  // The player takes the game back from Orion: stop renewing, then take the controller.
  function takeBack(){
    if(!orionPlay)return false;
    setOrion(false);
    if(!scriptedMode)return false;   // between turns: let the click or key do what it normally does
    el('takeover').click();
    return true;
  }
  setInterval(()=>{   // a pad press while Orion plays takes the game back (no input socket is polling then)
    if(!orionPlay || !scriptedMode)return;
    const pad=gamepad();
    if(pad && pad.buttons.some(b=>b?.pressed))takeBack();
  },100);
  async function showPolicy(){
    if(!sid)return;
    const current=sid, grantAsked=handoffGrant, state=await api(`sessions/${sid}`);
    if(current!==sid)return;
    if(state.state==='paused' && !(state.scripted||[]).length){scriptedMode=false;cognitivePort=null;handoffGrant=null;setOrion(false);}
    // Our grant ended (budget, Retro pause, or another controller such as leisure took the port):
    // leave handoff mode so blur releases the sockets and a click takes the game back.
    // Retro's status can lag the handoff, so the grant counts as gone only after a status showed it,
    // and only from a status requested after the handoff.
    if(scriptedMode && handoffGrant && handoffGrant===grantAsked){
      const mine=(state.scripted||[]).find(p=>p.port===handoffGrant.port);
      if(handoffGrant.epoch===null){if(mine){handoffGrant.epoch=mine.epoch;orionGrant={started_frame:mine.started_frame,duration_seconds:mine.duration_seconds};}}
      else if(!mine || mine.epoch!==handoffGrant.epoch){
        scriptedMode=false;cognitivePort=null;handoffGrant=null;
        // Renew only a turn that ran out its time. Anything else (another window took the
        // controller, a stop) ends Orion's play: never take the game back from someone.
        const fps=state.fps||60.0988, g=orionGrant; orionGrant=null;
        const ranOut=g && Number.isFinite(state.frame) && Number.isFinite(g.started_frame)
          && state.frame-g.started_frame >= (g.duration_seconds-5)*fps;
        if(orionPlay && ranOut)renewOrion();
        else if(orionPlay){setOrion(false);report('Someone else took the controller, so Orion stopped playing.');}
      }
    }
    el('policy').textContent=(state.scripted||[]).length
      ? state.scripted.map(p=>`Player ${p.port+1}: ${p.policy_id} (${p.cognition?'goal experiment':'scripted'}, ${p.duration_seconds}s limit)`).join(' · ')
      : 'No scripted controller active.';
    showDecisions(state);
    const policy=(state.scripted||[]).find(p=>p.port===cognitivePort && p.cognition);
    if(!policy)return;
    if(isZelda(policy.cognition)){planDecision(current,policy);return;}
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
  // Every 250 ms while Orion holds the controller (Link stands still from the end of one goal until
  // the next is planned; a 1 s poll added up to 1 s to every decision), else once a second.
  let pollingPolicy=false, lastPolicyPoll=0;
  setInterval(async()=>{if(document.hidden || pollingPolicy)return;
    const now=performance.now();
    if(!(scriptedMode && cognitivePort!==null) && now-lastPolicyPoll<1000)return;
    pollingPolicy=true;lastPolicyPoll=now;
    try{await showPolicy();}catch(_){el('policy').textContent='Controller status unavailable.';}
    finally{pollingPolicy=false;}},250);
  // --- Orion's play: Zelda goals, atlas and solo leisure (spec 2026-10-22, D/E) ------------------
  const waitingText = reason => 'Orion is waiting: '+String(reason||'GPU busy').replace(/^shed:/,'').replace(/_/g,' ');
  const goalText = g => !g ? 'wait' : g.kind+(g.dir?' '+g.dir:'')+(g.tile?` (${g.tile.x},${g.tile.y})`:'')+(g.target!==undefined?' '+g.target:'');
  const predictionText = p => !p ? 'none' : p.kind+(p.value!==undefined && p.value!==null?' '+p.value:'');
  // Co-play: the browser asks the Hub planner once per decision. No goal on failure; the motor waits.
  function planDecision(current,policy){
    const d=policy.cognition.attempt;
    if(!d || planning || plannedDecision===d.decision_id || performance.now()<planRetryAt)return;
    planning=true;const decisionId=d.decision_id, epoch=transportEpoch, currentConnection=connection;
    void api(`sessions/${current}/plan`,{connection_id:connection,generation,port:policy.port,expected_epoch:policy.epoch})
      .then(()=>{plannedDecision=decisionId;planRetryAt=0;})
      .catch(err=>{
        if(epoch!==transportEpoch || connection!==currentConnection)return;
        const b=err.body||{};
        if(b.detail==='gpu_pool_unavailable' && b.retry_after_sec){
          planRetryAt=performance.now()+b.retry_after_sec*1000;
          el('zgoal').textContent=`${waitingText(b.reason)}. Orion holds still and asks again in ${Math.round(b.retry_after_sec)} s. Take this controller any time.`;
        } else if(b.detail==='observation_unknown' || b.detail==='decision_pending'){planRetryAt=performance.now()+1000;}
        else {planRetryAt=performance.now()+5000;el('zgoal').textContent=`No goal: ${err.message}. Orion holds still; takeover remains available.`;}
      }).finally(()=>{planning=false;});
  }
  // Retro shows only the newest goal_finished row; the page keeps the last 5 per session.
  const ZELDA_POLICY='zelda-navigator-v1', ZELDA_PROFILE='zelda-1-ram-v1';
  const isZelda=c=>c?.policy_id===ZELDA_POLICY;
  let lastZelda=null, zeldaSession=false, finished=[], finishedSid=null;
  function showDecisions(state){
    zeldaSession=state.game_profile===ZELDA_PROFILE;
    if(finishedSid!==state.session_id){finished=[];finishedSid=state.session_id;}
    const p=(state.scripted||[]).find(x=>isZelda(x.cognition));
    if(!p){lastZelda=null;if(zeldaSession)el('play').hidden=false;return;}
    lastZelda=p.cognition;el('play').hidden=false;
    const g=p.cognition.goal, done=p.cognition.last_finished;
    if(done?.decision_id && !finished.some(h=>h.decision_id===done.decision_id)){finished.push(done);finished=finished.slice(-5);}
    if(g){el('zgoal').textContent=`Goal: ${goalText(g.goal)}. Why: ${g.rationale}`;
      el('zprediction').textContent=`Prediction: ${predictionText(g.prediction)} · budget ${g.budget_seconds} s`;}
    const rows=finished;
    el('verdicts').replaceChildren(...rows.map(h=>{const li=document.createElement('li');li.className=h.verdict||'unknown';
      li.textContent=`${goalText(h.goal)} → predicted ${predictionText(h.prediction)}: ${h.verdict||'unknown'}${h.outcome?' ('+h.outcome+')':''}`;return li;}));
  }
  function renderAtlas(atlas){
    // Retro's GET /sessions/{id}/atlas: {level, room, map{"level:room": {visits, exits, caves, deaths}}, ...}
    const level=Number.isInteger(atlas?.level)?atlas.level:0, here=Number.isInteger(atlas?.room)?atlas.room:null;
    const rooms=atlas?.map||{}, cells=[];
    for(let id=0;id<128;id++){
      const r=rooms[`${level}:${id}`], c=document.createElement('div');c.className='room';c.dataset.room=String(id);
      if(r){c.classList.add('known');const x=r.exits||{};for(const d of 'NSEW')if(x[d]!==null && x[d]!==undefined)c.classList.add('x'+d);
        const caves=Number(r.caves)||0, deaths=r.deaths||0;
        if(deaths)c.classList.add('death');
        c.textContent=(caves?'c':'')+(deaths?'×'+(deaths>1?deaths:''):'');
        c.title=`${level}:${id} · visits ${r.visits||0} · deaths ${deaths} · caves ${caves}`;}
      if(here===id){c.classList.add('current');c.setAttribute('aria-current','location');}
      cells.push(c);
    }
    el('atlas').replaceChildren(...cells);
    const known=Number.isInteger(atlas?.rooms_known)?atlas.rooms_known:Object.keys(rooms).length;
    el('atlasInfo').textContent=`${known} rooms known${level?` (showing level ${level})`:''}. Built only from what Orion saw. Bright edge: exit used. c: cave. ×: deaths. Outlined: current room.`;
  }
  // Watching never takes a controller: media only. Clicking the game or Take this controller
  // acquires one, which preempts Orion's leisure session at Retro's arbiter.
  function watch(){
    // Never race a human's own connect: it would bump the transport epoch under it.
    if(watching || input || media || !sid || document.hidden || connecting || wanted || healPending)return;
    watching=true;const epoch=++transportEpoch;
    lastPictureAt=performance.now();pictureReady=false;mediaHeals=0;mediaWindow=lastPictureAt;
    openMedia(epoch);
  }
  let atlasAt=0, pollingPlay=false;
  async function pollPlay(){
    const l=await api('leisure');
    const live=['starting','playing','waiting'].includes(l.state);
    el('leisure').classList.toggle('waiting',l.state==='waiting');
    if(live){
      el('play').hidden=false;
      const mins=Math.max(0,Math.round((Date.now()/1000-(l.started_at||Date.now()/1000))/60));
      el('leisure').textContent=l.state==='waiting'
        ? `${waitingText(l.waiting?.reason)} (retry in ${Math.round(l.waiting?.retry_after_sec||0)} s). Solo play, ${mins} min in.`
        : `Orion is playing alone (solo leisure), ${mins} min in, ${l.decisions||0} goals so far. Click the game or Take this controller to play; that ends Orion's session.`;
      if(l.session_id && l.session_id!==sid && !lease && !connecting && !healPending && !scriptedMode){
        ++intent;closeConnections();sid=l.session_id;activeSystem='nes'; // leisure currently supports NES Zelda only
      }
      if(l.session_id===sid && !lease && !input)watch();
      if(l.last_goal && !lastZelda){el('zgoal').textContent=`Goal: ${goalText(l.last_goal.goal)}. Why: ${l.last_goal.rationale}`;
        el('zprediction').textContent=`Prediction: ${predictionText(l.last_goal.prediction)}`;}
    } else if(l.state==='stale'){
      el('leisure').textContent='Orion\'s solo session stopped reporting; it is not running.';
    } else if(l.state==='ended' && l.end_reason){
      el('leisure').textContent=`Orion's last solo session ended: ${String(l.end_reason).replace(/_/g,' ')}.`;
    } else if(l.state==='refused' && l.reason){
      el('leisure').textContent=`Orion did not start solo play: ${String(l.reason).replace(/_/g,' ')}.`;
    }
    if(sid && (live || lastZelda || zeldaSession) && performance.now()-atlasAt>5000){
      atlasAt=performance.now();
      try{renderAtlas(await api(`sessions/${sid}/atlas`));el('play').hidden=false;}
      catch(err){el('atlasInfo').textContent='Atlas unavailable: '+err.message.replace(/_/g,' ')+'.';}
    }
  }
  setInterval(async()=>{if(document.hidden || pollingPlay)return;pollingPlay=true;
    try{await pollPlay();}catch(_){/* status is optional; play never depends on it */}
    finally{pollingPlay=false;}},2000);
  action('start',async()=>{focusGame();keys.add('Enter');send();setTimeout(()=>keys.delete('Enter'),150);});
  el('mute').onchange=()=>{if(gain)gain.gain.value=el('mute').checked?0:1;};
  el('port').onchange=neutral;
  el('tribute64').checked=tribute64;
  el('tribute64').onchange=()=>{tribute64=el('tribute64').checked;
    try{localStorage.setItem('retro_tribute64',tribute64?'1':'0');}catch(_){}padStatus();};
  // While the game has the keyboard, keys that scroll a page do nothing else. Before
  // this, unmapped PageUp/PageDown/Home/End/Space fell through to the browser and
  // scrolled the Hub page (the Retro frame is taller than the window) out from under
  // the game. In fullscreen, mapped keys reach the game even if focus slipped off it.
  const stage=el('stage'), SCROLL_KEYS=new Set(['PageUp','PageDown','Home','End','Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight']);
  const field=t=>t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement;
  // F7 saves and F10 loads the newest save into the running game, anywhere on the page
  // except while typing. Both keys have browser defaults (caret browsing, the menu bar),
  // so swallow them; with a modifier (Shift+F10 is the context menu) leave them alone.
  const typing=t=>t instanceof HTMLTextAreaElement || t?.isContentEditable ||
    (t instanceof HTMLInputElement && !['checkbox','radio','button','submit','reset','range','color','file'].includes(t.type));
  const HOTKEYS={F7:()=>el('save').click(),F10:()=>{el('saves').selectedIndex=0;hotkeyLoad=true;el('load').click();}};
  function runHotkey(code) {   // keyboard F7/F10 and the pad's Select chords
    if(!el('saveBar').hidden && !el('save').disabled && !el('load').disabled)HOTKEYS[code]();
  }
  document.addEventListener('keydown',e=>{
    if(HOTKEYS[e.code] && !typing(e.target) && !(e.shiftKey || e.ctrlKey || e.altKey || e.metaKey)){
      e.preventDefault();
      // Wait out a save or load in flight, so F10 never loads the save F7 is replacing.
      if(!e.repeat)runHotkey(e.code);
      return;
    }
    // T opens the chat box (no console in the key map uses T). Never while typing somewhere.
    if(e.code===CHAT_KEY && !chatOpen && !typing(e.target) && !(e.shiftKey || e.ctrlKey || e.altKey || e.metaKey)){
      e.preventDefault();if(!e.repeat)openChat();return;}
    if(chatOpen && e.code==='Escape' && e.target===el('chatInput')){e.preventDefault();closeChat();return;}
    if(orionPlay && scriptedMode && gameKey(e.code) && !field(e.target) && !e.target?.closest?.('button,summary,a,select')){
      e.preventDefault();takeBack();return;}
    if(document.activeElement!==screen && (document.fullscreenElement!==stage || field(e.target)))return;
    if(gameKey(e.code) || SCROLL_KEYS.has(e.code))e.preventDefault();
    if(gameKey(e.code) && !keys.has(e.code)){keys.add(e.code);send();}
  },{capture:true});
  action('fullscreen',async()=>{
    if(document.fullscreenElement){await document.exitFullscreen();return;}
    if(!stage.requestFullscreen || !document.fullscreenEnabled)throw Error('Fullscreen is blocked here. Open /retro directly.');
    await stage.requestFullscreen({navigationUI:'hide'});
  });
  el('exitFullscreen').onclick=()=>{if(document.fullscreenElement)void document.exitFullscreen().catch(()=>{});};
  document.addEventListener('fullscreenchange',()=>{
    el('fullscreen').textContent=document.fullscreenElement?'Exit fullscreen':'Fullscreen';focusGame();});
  window.addEventListener('keyup',e=>{keys.delete(e.code);send();});
  // Blur releases a human's held keys. A handoff holds no keys and the game is still on screen,
  // so blur keeps it: closing the socket would end Orion's turn (Retro cancels a grant whose
  // sponsor disconnects). A hidden tab or closed page still ends the handoff: nobody can watch.
  window.addEventListener('blur',()=>{away=true;if(!scriptedMode)neutral();});window.addEventListener('pagehide',neutral);
  window.addEventListener('gamepaddisconnected',()=>{padStatus();if(!scriptedMode)neutral();});
  window.addEventListener('gamepadconnected',padStatus);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){away=true;neutral();}else{away=false;heal();}});
  window.addEventListener('focus',()=>{away=false;heal();});
  screen.addEventListener('pointerdown',()=>{
    if(chatOpen)closeChat();   // a click on the game means play: the click below takes the controller
    away=false;if(takeBack())return;if(scriptedMode)return;
    if(sid){wanted=true;heals=0;if(!connecting && !acquiring && !lease)void connect().catch(err=>report(err.message));}
  });
  // Who plays (game company design, patch 1). The server knows only that buttons move; it never
  // guesses a name. The page asks once per sitting and tells the server what it was told
  // ('juniper' or 'guest'). No answer is a valid state ("Not now"): the record then says "someone".
  let sittingAsked=null, pollingSitting=false;
  try{sittingAsked=sessionStorage.getItem('retroSittingAsked');}catch(_){}   // a reload does not ask again
  const settle=id=>{
    sittingAsked=id;el('whoPlays').hidden=true;
    try{sessionStorage.setItem('retroSittingAsked',id);}catch(_){}
    focusGame();                                                             // the game keeps the keys
  };
  // Orion's look (game company design, patch 2). The server says whether Orion is watching and, if not,
  // why: the text is the server's, never composed here, so the page cannot claim more than the record.
  function renderWatch(w,poll=false){
    const box=el('watch');
    if(!w || !w.enabled){box.hidden=true;return;}
    box.hidden=false;
    el('watchText').textContent=(w.status && w.status.text)||'';
    // Three positions (spec section 5): off, quiet (looks, no lines), on (watch and talk).
    const sw=el('watchSwitch');
    if(poll && (sw.disabled || document.activeElement===sw))return;   // never flip it under a choice in progress
    sw.value=['off','quiet','on'].includes(w.juniper_mode)?w.juniper_mode:'on';sw.dataset.mode=sw.value;
  }
  el('watchSwitch').onchange=async()=>{
    const sw=el('watchSwitch');sw.disabled=true;
    try{renderWatch((await api('sitting/watch',{mode:sw.value})).watch);}
    catch(err){sw.value=sw.dataset.mode||'on';report(err.message);}
    finally{sw.disabled=false;focusGame();}                              // the game keeps the keys
  };
  async function pollSitting(){
    const got=await api('sitting');
    renderWatch(got.watch,true);
    const s=got.sitting;
    chatSitting(s);
    el('whoPlays').dataset.sitting=s?.sitting_id||'';
    // Only the page that holds the controller is asked: a second screen that only watches is not.
    el('whoPlays').hidden=!(s && lease && !s.player_stated && s.sitting_id!==sittingAsked);
  }
  for(const [id,player] of [['whoJuniper','juniper'],['whoGuest','guest']])action(id,async()=>{
    const asked=el('whoPlays').dataset.sitting;
    try{await api('sitting/player',{player,sitting_id:asked});   // an answer to an older sitting is refused
      if(asked && asked===chat.sitting)chat.player=player;}        // later chat messages use it; sent ones keep theirs
    finally{settle(asked);}
  });
  action('whoLater',async()=>settle(el('whoPlays').dataset.sitting));
  setInterval(()=>{
    if(document.hidden || pollingSitting)return;
    pollingSitting=true;pollSitting().catch(()=>{}).finally(()=>{pollingSitting=false;});
  },5000);

  // --- The chat box (game company design, section 5) ------------------------------------------------
  // A panel over the lower part of the screen, inside #stage so fullscreen shows it, talking on the
  // Hub chat socket (/ws). Who typed decides the session: Juniper stated -> her main chat thread
  // (localStorage.orion_sid); a guest or nobody stated -> a page-local "retro-guest-<sitting_id>".
  // Each message keeps the session it was sent with; a later answer to "Who's playing?" changes only
  // later messages. The line strip shows Orion's newest line or reply, then fades.
  const CHAT_KEY='KeyT', STRIP_FADE_MS=15000;
  const chat={sitting:null, player:null, socket:null, hello:null, retry:0, retryTimer:null,
    pausedGame:false, pending:0, stripTimer:null, guest:'retro-guest-page-'+Math.random().toString(36).slice(2,10)};
  function chatSitting(s){
    const id=s?.sitting_id||null;
    if(id && id!==chat.sitting){
      if(chat.sitting)el('chatLog').replaceChildren();   // the panel holds this sitting's lines and messages
      chat.sitting=id;
    }
    if(s && s.player_stated!==undefined)chat.player=s.player_stated||null;
  }
  // Juniper's thread, created the way the main chat page does (GET /api/session) when absent.
  async function juniperSid(){
    let sid=null;try{sid=localStorage.getItem('orion_sid');}catch(_){}
    if(sid)return sid;
    const r=await fetch('/api/session',{headers:{}});const d=await r.json();
    if(!d.session_id)throw Error('No chat session.');
    try{localStorage.setItem('orion_sid',d.session_id);}catch(_){}
    return d.session_id;
  }
  async function chatSid(){
    return chat.player==='juniper' ? juniperSid() : 'retro-guest-'+(chat.sitting||chat.guest.slice('retro-guest-'.length));
  }
  // The contract's typed message (Browser <-> server): Orion mode, no speech, the retro surface.
  function chatMessage(text, sessionId, sittingId){
    return {text_input:text, mode:'orion', disable_tts:true, session_id:sessionId,
      surface_context:{surface:'retro', input_modality:'typed', retro_sitting_id:sittingId}};
  }
  const thumbURL=lineId=>`lines/${encodeURIComponent(lineId)}/thumb`;
  const wsChatURL=()=>`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`;
  function hello(sessionId){
    if(chat.socket?.readyState!==WebSocket.OPEN || chat.hello===sessionId)return;
    chat.socket.send(JSON.stringify({type:'session_hello', session_id:sessionId}));chat.hello=sessionId;
  }
  // Reconnect with backoff (1 s doubling to 30 s). Nothing here touches focus.
  function chatConnect(){
    clearTimeout(chat.retryTimer);chat.retryTimer=null;
    if(document.hidden)return;
    let socket;try{socket=new WebSocket(wsChatURL());}catch(_){chatRetry();return;}
    chat.socket=socket;chat.hello=null;
    socket.onopen=()=>{chat.retry=0;el('chatInfo').textContent='';
      void chatSid().then(hello).catch(()=>{});};
    socket.onclose=()=>{if(chat.socket!==socket)return;chat.socket=null;chat.hello=null;
      if(chat.pending){chat.pending=0;chatEntry('system','The chat connection dropped; the reply was lost. Send again.');}
      chatRetry();};
    socket.onmessage=e=>{let d;try{d=JSON.parse(e.data);}catch(_){return;}if(d && typeof d==='object')chatFrame(d);};
  }
  function chatRetry(){
    if(chat.retryTimer)return;
    const wait=Math.min(30000,1000*2**chat.retry++);
    el('chatInfo').textContent=`Chat reconnecting in ${Math.round(wait/1000)} s.`;
    chat.retryTimer=setTimeout(chatConnect,wait);
  }
  document.addEventListener('visibilitychange',()=>{if(!document.hidden && !chat.socket && !chat.retryTimer)chatConnect();});
  function chatFrame(d){
    if(d.kind==='orion_outreach'){
      const text=String(d.llm_response||'').trim();if(!text)return;
      if(d.source==='retro_commentary' && d.line_id){   // a line: strip with thumbs, and the panel
        chatEntry('orion',text,{lineId:String(d.line_id)});strip(text,String(d.line_id));
      } else if(chat.player==='juniper') chatEntry('orion',text,{note:'Orion, unprompted'});   // other outreach: panel only
      // Orion's other messages are to Juniper: a guest (or nobody stated) at the game never sees them here.
      return;
    }
    if(d.type==='final'){
      const text=String(d.llm_response||'').trim();chat.pending=Math.max(0,chat.pending-1);
      if(text){chatEntry('orion',text);strip(text,null);}
      return;
    }
    if(d.type==='turn_error' || d.type==='turn_deferred'){
      chat.pending=Math.max(0,chat.pending-1);
      chatEntry('system',d.type==='turn_error'?`Orion could not answer (${d.error||d.phase||'turn failed'}).`
        :`Orion put this off (${d.reason||'not now'}).`);
    }
  }
  function thumbButtons(lineId){
    const box=document.createElement('span');box.className='thumbs';box.dataset.line=lineId;
    for(const [thumb,label] of [['up','Thumb up'],['down','Thumb down']]){
      const b=document.createElement('button');b.type='button';b.className='thumb';b.dataset.thumb=thumb;
      b.textContent=thumb==='up'?'+':'−';b.title=label;b.setAttribute('aria-label',label);b.setAttribute('aria-pressed','false');
      b.onclick=()=>void sendThumb(lineId,thumb);box.append(b);
    }
    const state=document.createElement('small');state.className='thumbState';box.append(state);
    return box;
  }
  async function sendThumb(lineId,thumb){
    const boxes=()=>document.querySelectorAll(`.thumbs[data-line="${CSS.escape(lineId)}"]`);
    for(const box of boxes())for(const b of box.querySelectorAll('button'))b.disabled=true;
    try{
      const r=await api(thumbURL(lineId),{thumb});
      for(const box of boxes()){
        for(const b of box.querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.thumb===thumb));
        box.dataset.stored=thumb;
        box.querySelector('.thumbState').textContent=`Saved: thumb ${thumb}${r.thumb_by==='guest'?' (guest)':''}.`;
      }
    }catch(err){
      for(const box of boxes())box.querySelector('.thumbState').textContent=err.status===404?'That line is gone.':'Not saved: '+err.message;
    }finally{
      for(const box of boxes())for(const b of box.querySelectorAll('button'))b.disabled=false;
      focusGame();   // a click on a thumb never keeps the keyboard from the game
    }
  }
  function chatEntry(who,text,opts={}){
    const row=document.createElement('div');row.className='chatRow '+who;
    const name=document.createElement('b');
    name.textContent=who==='orion'?'Orion':who==='system'?'':(opts.player==='juniper'?'Juniper':'Guest');
    const body=document.createElement('span');body.textContent=text;
    row.append(name,' ',body);
    if(opts.note){const n=document.createElement('small');n.textContent=` (${opts.note})`;row.append(n);}
    if(opts.lineId){row.dataset.line=opts.lineId;row.append(' ',thumbButtons(opts.lineId));}
    const log=el('chatLog');log.append(row);
    while(log.children.length>200)log.firstChild.remove();
    log.scrollTop=log.scrollHeight;
  }
  function strip(text,lineId){
    const box=el('chatStrip');clearTimeout(chat.stripTimer);
    el('chatStripText').textContent=text;
    const old=box.querySelector('.thumbs');if(old)old.remove();
    if(lineId){
      const thumbs=thumbButtons(lineId), seen=document.querySelector(`#chatLog .thumbs[data-line="${CSS.escape(lineId)}"]`);
      if(seen?.dataset.stored){thumbs.dataset.stored=seen.dataset.stored;
        for(const b of thumbs.querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.thumb===seen.dataset.stored));
        thumbs.querySelector('.thumbState').textContent=seen.querySelector('.thumbState').textContent;}
      box.append(thumbs);
    }
    box.dataset.line=lineId||'';box.hidden=false;box.classList.remove('fading');
    chat.stripTimer=setTimeout(()=>{box.classList.add('fading');chat.stripTimer=setTimeout(()=>{box.hidden=true;},1000);},STRIP_FADE_MS);
  }
  async function sendChat(){
    const input=el('chatInput'), text=input.value.trim();
    if(!text)return;
    if(chat.socket?.readyState!==WebSocket.OPEN){chatEntry('system','Not connected to Orion yet; try again in a moment.');return;}
    const player=chat.player, sittingId=chat.sitting;   // fixed now: a later flip never rewrites this message
    let sessionId;
    try{sessionId=await chatSid();}catch(err){chatEntry('system','No chat session: '+err.message);return;}
    if(chat.socket?.readyState!==WebSocket.OPEN){chatEntry('system','Not connected to Orion yet; try again in a moment.');return;}
    hello(sessionId);
    chat.socket.send(JSON.stringify(chatMessage(text,sessionId,sittingId)));
    chat.pending++;input.value='';
    chatEntry('you',text,{player});
  }
  function openChat(){
    if(chatOpen){el('chatInput').focus();return;}
    chatOpen=true;el('chatPanel').hidden=false;stage.classList.add('chatting');el('chatToggle').setAttribute('aria-expanded','true');
    keys.clear();send();   // one neutral input: a button held while opening is released
    // A keyboard player cannot type and play at once: with no controller connected, pause.
    // With a controller, the controller keeps driving the game.
    if(lease && !scriptedMode && !gamepad()){chat.pausedGame=true;void pauseGame().then(()=>report('Paused while you type. Close the chat to keep playing.')).catch(err=>report(err.message));}
    el('chatInput').focus();
  }
  function closeChat(){
    if(!chatOpen)return;
    chatOpen=false;el('chatPanel').hidden=true;stage.classList.remove('chatting');el('chatToggle').setAttribute('aria-expanded','false');
    if(chat.pausedGame){chat.pausedGame=false;if(sid)void connect().catch(err=>report(err.message));}
    focusGame();
  }
  el('chatToggle').onclick=()=>chatOpen?closeChat():openChat();
  el('chatClose').onclick=closeChat;
  el('chatForm').onsubmit=e=>{e.preventDefault();void sendChat();};
  padStatus();
  // The chat socket opens once the page has loaded (or failed to), never ahead of the game's own sockets.
  void load().catch(err=>{el('player').hidden=false;report(err.message);}).finally(chatConnect);
})();

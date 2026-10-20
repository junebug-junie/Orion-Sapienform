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
    transportEpoch++; clearInterval(timer); keys.clear(); lease=null; outstanding.clear();
    if (input) { input.onclose=null; input.close(); } if (media) { media.onclose=null; media.close(); }
    input=null;media=null; pendingPacket=null; flushAudio();
  }
  function neutral() { closeConnections(); report('Controller released. Use Connect / resume to return.'); }
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
      neutral();report('Connection stalled. Use Connect / resume.');return;
    }
    outstanding.set(seq+1,performance.now());input.send(JSON.stringify({version:1,session_id:sid,generation,...lease,seq:++seq,mask:mask()}));
  }
  async function grant(action='acquire') {
    const epoch=transportEpoch;
    const result=await api(`sessions/${sid}/control`,{connection_id:connection,port:Number(el('port').value),action});
    if(epoch!==transportEpoch || document.hidden)return;
    lease=result;
    seq=0; outstanding.clear();clearInterval(timer);timer=setInterval(send,50);send();screen.focus();
  }
  const wsURL = path => `${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/retro/${path}`;
  async function connect() {
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
      report('Connected.');return;
    }
    closeConnections(); const epoch=transportEpoch; await sound();
    if(epoch!==transportEpoch || document.hidden)return;
    if(!sid) { const state=await api('health/ready');sid=state.session?.session_id; }
    if(epoch!==transportEpoch || document.hidden)return;
    if(!sid)throw Error('Start a new game first.');
    await api(`sessions/${sid}/resume`,{});
    if(epoch!==transportEpoch || document.hidden)return;
    input=new WebSocket(wsURL(`sessions/${sid}/input`));
    input.onclose=()=>{if(epoch===transportEpoch)neutral();};
    input.onmessage=async e=>{
      if(epoch!==transportEpoch)return;
      const msg=JSON.parse(e.data);
      if(msg.connection_id) { connection=msg.connection_id;generation=msg.generation;
        try { await grant(); } catch(err) {report(err.message+' — choose another player or take this controller.');}
      } else if(msg.applied){
        if(!lease || msg.applied.epoch!==lease.epoch || msg.applied.port!==lease.port)return;
        outstanding.delete(msg.applied.seq);screen.dataset.appliedFrame=msg.applied.frame;
      } else {if(msg.epoch!==undefined && (!lease || msg.epoch!==lease.epoch || msg.port!==lease.port))return;
        // A newer input replaced this one before its frame; the lease is still live.
        if(msg.rejected==='superseded' || msg.rejected==='expired_in_queue'){outstanding.delete(msg.seq);return;}
        lease=null;outstanding.clear();clearInterval(timer);report('Control interrupted: '+(msg.rejected||msg.unknown||'connection unavailable')+'. Use Connect / resume.');}

    };
    media=new WebSocket(wsURL(`sessions/${sid}/media`)); media.binaryType='arraybuffer';
    media.onclose=()=>{if(epoch===transportEpoch){neutral();report('Video connection closed. Use Connect / resume.');}};
    media.onmessage=e=>{pendingPacket={data:e.data,epoch};void drain();};
    report('Connected. Click the screen, then press Start.');
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
  async function load(){const games=await api('games');el('game').replaceChildren(...games.map(g=>{const o=document.createElement('option');o.value=g.game_id;o.textContent=g.title;o.dataset.ports=JSON.stringify(g.ports);return o;}));el('game').value=games.some(g=>g.game_id==='spacegulls')?'spacegulls':games[0]?.game_id;const state=await api('health/ready');sid=state.session?.session_id;
    el('player').hidden=false;
    if(state.session && ['running','paused','starting'].includes(state.session.state)){el('game').value=state.session.game_id;report('Game '+state.session.state+'. Connect / resume to control it.');}else report('Choose a game.');}
  const action = (id, fn) => el(id).onclick=async()=>{el(id).disabled=true;try{await fn();}catch(err){report(err.message);}finally{el(id).disabled=false;}};
  action('new',async()=>{const epoch=transportEpoch;await sound();if(epoch!==transportEpoch || document.hidden)return;const s=await api('sessions',{game_id:el('game').value,idempotency_key:Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,"0")).join("")});sid=s.session_id;if(epoch===transportEpoch && !document.hidden)await connect();});
  action('join',connect);
  action('pause',async()=>{if(sid)await api(`sessions/${sid}/pause`,{});neutral();});
  action('stop',async()=>{if(sid)await api(`sessions/${sid}/stop`,{});neutral();sid=null;report('Game stopped.');});
  action('takeover',()=>grant('takeover'));
  action('start',async()=>{screen.focus();keys.add('Enter');send();setTimeout(()=>keys.delete('Enter'),150);});
  el('mute').onchange=()=>{if(gain)gain.gain.value=el('mute').checked?0:1;};
  el('port').onchange=neutral;
  screen.addEventListener('keydown',e=>{if(bits[e.code]){e.preventDefault();if(!keys.has(e.code)){keys.add(e.code);send();}}});
  window.addEventListener('keyup',e=>{keys.delete(e.code);send();});
  window.addEventListener('blur',neutral);window.addEventListener('pagehide',neutral);window.addEventListener('gamepaddisconnected',neutral);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)neutral();});
  void load().catch(err=>report(err.message));
})();

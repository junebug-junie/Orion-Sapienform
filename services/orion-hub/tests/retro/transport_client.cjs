// Executes the actual page asset with deterministic clocks, DOM and socket doubles.
const {test}=require('node:test'), assert=require('node:assert/strict');
const fs=require('node:fs'), vm=require('node:vm'), path=require('node:path');

function page(){
  let now=100, next=0;const timers=new Map(), elements=new Map(), sockets=[], requests=[];
  const element=id=>{
    if(!elements.has(id))elements.set(id,{value:'0',checked:false,hidden:false,disabled:false,dataset:{},
      textContent:'',events:{},setAttribute(){},getContext:()=>({createImageData:(w,h)=>({data:new Uint8Array(w*h*4)}),putImageData(){}}),addEventListener(name,fn){this.events[name]=fn;},focus(){},
      classList:{toggle(){}},selectedOptions:[{dataset:{}}]});
    return elements.get(id);
  };
  class Socket{
    static OPEN=1;
    constructor(url){this.url=url;this.readyState=1;this.bufferedAmount=0;this.sent=[];this.closed=false;sockets.push(this);}
    send(data){this.sent.push(JSON.parse(data));}
    close(){this.closed=true;this.readyState=3;}
    fail(){this.close();this.onclose?.();}
  }
  let responder=()=>new Promise(()=>{});
  const context=vm.createContext({console,Map,Set,Blob,Response,DecompressionStream,TextDecoder,AbortController,
    Uint8Array,DataView,performance:{now:()=>now},WebSocket:Socket,
    location:{protocol:'http:',host:'fixture'},navigator:{getGamepads:()=>[]},
    HTMLInputElement:class{},HTMLSelectElement:class{},HTMLTextAreaElement:class{},
    window:{addEventListener(){}},document:{getElementById:element,addEventListener(){},hidden:false,
      hasFocus:()=>true,activeElement:null},
    setTimeout:(fn,ms)=>{timers.set(++next,{fn,at:now+ms});return next;},
    clearTimeout:id=>timers.delete(id),
    setInterval:(fn,ms)=>{timers.set(++next,{fn,at:now+ms,interval:ms});return next;},
    clearInterval:id=>timers.delete(id),
    fetch:async(url,options)=>{requests.push({url,options});return responder(url,options);},
  });
  let source=fs.readFileSync(path.join(__dirname,'../../static/js/retro-panel.js'),'utf8');
  source=source.replace(/\}\)\(\);\s*$/, 'globalThis.probe={run:code=>eval(code)};})();');
  vm.runInContext(source,context);
  const run=code=>context.probe.run(code);
  run("sound=async()=>{};sid='session-a';generation='generation-a';wanted=true;conditionalResume=true;lease={lease:'test',epoch:1,port:0};input=new WebSocket('input');pictureReady=true;lastPictureAt=performance.now();");
  return {run,sockets,requests,elements,context,
    respond:fn=>responder=async(url,options)=>{const [status,data]=await fn(url,options);return {ok:status<400,status,json:async()=>data};},
    async tick(ms){const end=now+ms;while(true){
      const due=[...timers].filter(([,t])=>t.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];
      if(!due)break;const [id,t]=due;now=t.at;if(t.interval)t.at+=t.interval;else timers.delete(id);
      t.fn();for(let n=0;n<12;n++)await Promise.resolve();
    }now=end;for(let n=0;n<12;n++)await Promise.resolve();},
  };
}

test('media disconnect preserves input, sends neutral, then recovers only media',async()=>{
  const h=page();h.run('openMedia(transportEpoch)');const input=h.sockets[0], media=h.sockets[1];
  media.fail();assert.equal(input.closed,false);assert.equal(input.sent.at(-1).mask,0);
  await h.tick(300);assert.equal(h.sockets.length,3);assert.equal(h.sockets[2].url.endsWith('/media'),true);
  assert.equal(input.closed,false);assert.equal(h.requests.filter(r=>r.url.endsWith('/resume')).length,0);
});

test('silent media neutralizes held keys and closes input within three seconds',async()=>{
  const h=page();h.run("keys.add('ArrowRight');send()");assert.equal(h.sockets[0].sent.at(-1).mask,128);
  await h.tick(1100);h.run('outstanding.clear();send()');assert.equal(h.sockets[0].sent.at(-1).mask,0);
  await h.tick(2000);assert.equal(h.sockets[0].closed,true);
});

test('co-play sponsor is bounded on picture loss and never automatically reacquired',async()=>{
  const h=page();h.run('lease=null;scriptedMode=true;wanted=false;orionPlay=true');
  await h.tick(3100);assert.equal(h.sockets[0].closed,true);
  await h.tick(1000);assert.equal(h.requests.some(r=>r.url.endsWith('/control') || r.url.endsWith('/resume')),false);
  assert.equal(h.run('orionPlay'),false);
});

test('pause and stop cancel queued recovery and close surviving sockets',async()=>{
  for(const action of ['pause','stop']){
    const h=page();h.respond(()=>[200,{state:'paused'}]);
    h.run('transportLost("closed")');await h.elements.get(action).onclick();
    await h.tick(600);assert.equal(h.sockets.length,1);
    assert.equal(h.requests.some(r=>r.url.endsWith('/resume') || r.url.endsWith('/control')),false);
  }
});

test('session replacement, hidden page, or lost focus cancels queued controller recovery',async()=>{
  for(const change of ["sid='session-b'",'document.hidden=true','document.hasFocus=()=>false']){
    const h=page();h.run('transportLost("closed")');h.run(change);await h.tick(600);
    assert.equal(h.sockets.length,1);assert.equal(h.requests.length,1);
  }
});

test('old media retry cannot reopen after explicit pause',async()=>{
  const h=page();h.respond(()=>[200,{}]);h.run('openMedia(transportEpoch)');h.sockets[1].fail();
  await h.elements.get('pause').onclick();await h.tick(600);assert.equal(h.sockets.length,2);
});

test('hung Pause/Stop immediately releases input and prevents delayed grant installation',async()=>{
  for(const action of ['pause','stop']){
    const h=page();let controlReply;
    h.respond(url=>url.endsWith('/control')?new Promise(resolve=>controlReply=resolve):new Promise(()=>{}));
    const grant=h.run('grant()');await Promise.resolve();
    void h.elements.get(action).onclick(); // HTTP intentionally never answers
    assert.equal(h.sockets[0].closed,true);
    controlReply([200,{lease:'obsolete',epoch:2,port:0}]);await grant;
    await h.tick(4000);assert.equal(h.run('lease'),null);assert.equal(h.sockets.length,1);
  }
});

test('acquire retries an automatic disconnect pause once without takeover',async()=>{
  const h=page();let controls=0;
  h.respond((url)=>url.endsWith('/control')?(++controls===1?[409,{detail:'session_paused:disconnect'}]:[200,{lease:'new',epoch:2,port:0}]):[200,{state:'running'}]);
  await h.run('grant()');
  assert.equal(controls,2);assert.equal(h.requests.filter(r=>r.url.endsWith('/resume')).length,1);
  assert.equal(JSON.parse(h.requests.find(r=>r.url.endsWith('/resume')).options.body).expected_pause_reason,'disconnect');
  assert(h.requests.filter(r=>r.url.endsWith('/control')).every(r=>JSON.parse(r.options.body).action==='acquire'));
});

test('acquire never resumes explicit pause or takes another human controller',async()=>{
  for(const reason of ['session_paused:pause','port_owned']){
    const h=page();h.respond(()=>[409,{detail:reason}]);
    await assert.rejects(h.run('grant()'),new RegExp(reason));
    assert.equal(h.requests.filter(r=>r.url.endsWith('/control')).length,1);
    assert.equal(h.requests.some(r=>r.url.endsWith('/resume')),false);
  }
});

test('automatic reconnect does not resume before the authoritative acquire response',async()=>{
  const h=page();h.respond(()=>[200,{}]);h.run('closeConnections()');await h.run('connect(true)');
  assert.equal(h.requests.some(r=>r.url.endsWith('/resume')),false);
  assert.equal(h.sockets.filter(s=>s.url.endsWith('/input')).length,1);
});

test('deliberate game click resumes after an explicit Pause',async()=>{
  const h=page();h.respond(()=>[200,{state:'running'}]);
  await h.elements.get('pause').onclick();
  h.elements.get('screen').events.pointerdown();await h.tick(0);
  assert.equal(h.requests.filter(r=>r.url.endsWith('/resume')).length,1);
  assert.equal(h.sockets.filter(s=>s.url.endsWith('/input')).length,1);
});

test('clicking while the controller grant is delayed cannot invalidate that grant',async()=>{
  const h=page();h.run('closeConnections();lease=null;wanted=true');h.respond(()=>[200,{}]);
  await h.run('connect(true)');
  const socket=h.sockets.find(s=>s.url.endsWith('/input'));
  let respond;
  h.respond(()=>new Promise(resolve=>respond=resolve));
  const receiving=socket.onmessage({data:JSON.stringify({connection_id:'owner',generation:'gen'})});
  await Promise.resolve();const finishGrant=respond, originalIntent=h.run('intent');
  h.elements.get('screen').events.pointerdown();await h.tick(0);
  finishGrant([200,{lease:'accepted',epoch:2,port:0}]);await receiving;
  assert.equal(h.run('intent'),originalIntent,'extra click invalidated pending controller acquisition');
  assert.equal(h.run('lease?.lease'),'accepted');
  assert.equal(h.requests.filter(r=>r.url.endsWith('/control')).length,1);
});

test('late normal acquisition cannot replace an explicit takeover',async()=>{
  const h=page(), replies=[];
  h.respond(url=>url.endsWith('/control')?new Promise(resolve=>replies.push(resolve)):[200,{}]);
  const old=h.run('grant()');await Promise.resolve();
  const takeover=h.elements.get('takeover').onclick();await h.tick(0);
  assert.equal(replies.length,2);
  replies[1]([200,{lease:'new-takeover',epoch:3,port:0}]);await takeover;
  replies[0]([200,{lease:'old-acquire',epoch:2,port:0}]);await old;
  assert.equal(h.run('lease.lease'),'new-takeover');
});

test('hung acquisition is aborted, releases control and cannot wedge connecting',async()=>{
  const h=page();h.run("lease=null;connection='owner'");let aborted=false;
  h.respond((url,options)=>url.endsWith('/control')?new Promise((_,reject)=>{
    options.signal.addEventListener('abort',()=>{aborted=true;reject(Error('aborted'));});
  }):[200,{state:'running'}]);
  const connecting=h.run('connect(true)');await h.tick(0);
  await h.tick(5100);await connecting;
  assert.equal(aborted,true);assert.equal(h.sockets[0].closed,true);
  assert.equal(h.run('acquiring'),null);assert.equal(h.run('connecting'),0);
  await h.tick(500);
  assert.equal(h.sockets.some(s=>s.url.endsWith('/input')),true,'timeout never retried controller connection');
});

test('old native service cannot receive an unsafe automatic resume',async()=>{
  const h=page();h.run('conditionalResume=false');h.respond(()=>[409,{detail:'session_paused:disconnect'}]);
  await assert.rejects(h.run('grant()'),/session_paused:disconnect/);
  assert.equal(h.requests.some(r=>r.url.endsWith('/resume')),false);
});

test('media credits are negotiated and acknowledged only after a current frame is presented',async()=>{
  const h=page();
  const header=Buffer.from(JSON.stringify({version:1,width:1,height:1,format:1,video_bytes:4,audio_frames:0,frame:30}));
  const prefix=Buffer.alloc(4);prefix.writeUInt32BE(header.length);
  h.context.packet=require('node:zlib').deflateSync(Buffer.concat([prefix,header,Buffer.alloc(4)]));
  h.run('mediaFlowControl=true;openMedia(transportEpoch)');const socket=h.sockets.at(-1);
  assert(socket.url.endsWith('/media?flow=ack-v1'));
  await h.run('draw({data:globalThis.packet,epoch:transportEpoch,socket:media,flow:true,sequence:3})');
  assert.equal(socket.sent[0].ack,3);assert.equal(h.elements.get('screen').dataset.frame,30);
  h.run('mediaFlowControl=false;openMedia(transportEpoch)');const legacy=h.sockets.at(-1);
  assert(legacy.url.endsWith('/media'));
  await h.run('draw({data:globalThis.packet,epoch:transportEpoch,socket:media,flow:false,sequence:1})');
  assert.equal(legacy.sent.length,0,'unnegotiated ACK would break an older gateway');
  h.run('mediaFlowControl=true;openMedia(transportEpoch)');
  h.run('globalThis.oldMedia=media;media=new WebSocket("replacement")');
  await h.run('draw({data:globalThis.packet,epoch:transportEpoch,socket:globalThis.oldMedia,flow:true,sequence:4})');
  assert.equal(h.run('globalThis.oldMedia.sent.length'),0,'old decoder acknowledged a replacement connection');
});

test('old asynchronous decoder failure cannot close a replacement transport or abandon its packet',async()=>{
  const h=page();
  h.run(`media=new WebSocket('old-media');
    globalThis.failDecode=null;globalThis.draws=0;
    draw=async()=>{if(++globalThis.draws===1)await new Promise((_,reject)=>globalThis.failDecode=reject);};
    pendingPacket={epoch:transportEpoch,socket:media};void drain();
    neutral();input=new WebSocket('new-input');media=new WebSocket('new-media');
    pendingPacket={epoch:transportEpoch,socket:media};globalThis.failDecode(Error('old decode failed'));`);
  await h.tick(0);
  assert.equal(h.sockets.find(s=>s.url==='new-input').closed,false);
  assert.equal(h.run('globalThis.draws'),2);
  assert.equal(h.run('decoding'),false);
});

test('replaced session or paused intent discards delayed acquisition',async()=>{
  for(const change of ["sid='session-b'",'++intent;closeConnections()']){
    const h=page();let reply;
    h.respond(()=>new Promise(resolve=>reply=resolve));
    const granting=h.run('grant()');await Promise.resolve();h.run(change);
    reply([200,{lease:'obsolete',epoch:2,port:0}]);await granting;
    assert.notEqual(h.run('lease?.lease'),'obsolete');
  }
});

/** Deterministic simulation. No clock, network, Convex or model calls here. */
export type Need = 'money' | 'rest' | 'social' | 'purpose';
export type Effect = {op: 'stock_delta'|'need_delta'|'fact_set'|'trust_delta'|'place_flag'|'belief_set'; target: string; value: number|string|boolean; source?: string};
export type Condition = {target: string; cmp: 'eq'|'gte'; value: string|number|boolean};
export type Quest = {id: string; title: string; participants: string[]; goal: Condition[]; steps: Condition[][]; deadline: number; status: 'open'|'succeeded'|'failed'|'abandoned'; reward: Effect[]; penalty: Effect[]; effect_error?:string};
export type Agent = {name: string; voice: string; needs: Record<Need,number>; home: string; work: string; location: string; schedule: {start:number; end:number}; weights: Record<string,number>; lyingRate: number};
export type State = {version:number; minute:number; seed:number; places:Record<string,{name:string;x:number;y:number}>; stock:Record<string,number>; agents:Record<string,Agent>; trust:Record<string,number>; facts:Record<string,string|number|boolean>; beliefs:Record<string,{value:string|number|boolean;confidence:number;source:string;learned_at:number}>; flags:Record<string,string|number|boolean>; quests:Record<string,Quest>};
export type Result = {id:string; title:string; effects:Effect[]};
export type Scene = {lines:{speaker:string;text:string}[]; results:string[]; why:string};
export type Event = {id:string; source:'sim'|'mirror'|'juniper'|'orion'; kind:string; title:string; effects:Effect[]; quest?:Quest; expires_at?:number};
export const PURPOSES = ['breeze','ask_favor','trade_news','persuade','deceive','quest_step'];
const needs = ['money','rest','social','purpose'];
const own = (o:object,k:string) => Object.prototype.hasOwnProperty.call(o,k);
export function assert(ok:unknown, message:string): asserts ok {if (!ok) throw new Error(message);}
export function clone<T>(value:T):T {return JSON.parse(JSON.stringify(value));}
export function draw(s:State):number {s.seed=(Math.imul(s.seed,1664525)+1013904223)>>>0; return s.seed/4294967296;}
const clamp=(n:number,lo=0,hi=1)=>Math.max(lo,Math.min(hi,n));
export function createState(cast:Record<string,Agent>, seed=1):State {
  const places = {square:{name:'Square',x:16,y:16}, diner:{name:'Diner',x:8,y:8}, workshop:{name:'Workshop',x:24,y:16}, studio:{name:'Studio',x:16,y:24}};
  const s:State={version:0,minute:0,seed:seed>>>0,places,stock:{'diner.coffee':24,'workshop.parts':5,'square.flyers':12},agents:clone(cast),trust:{},facts:{'shipment.cancelled':false,'generator.fixed':false},beliefs:{},flags:{},quests:{}};
  for(const [id,a] of Object.entries(s.agents)) {
    assert(own(places,a.home)&&own(places,a.work),'unknown home/work');
    for(const n of needs) assert(Number.isFinite(a.needs[n as Need])&&a.needs[n as Need]>=0&&a.needs[n as Need]<=1,'invalid need');
    assert(a.schedule.start>=0&&a.schedule.end<=24&&a.schedule.start<a.schedule.end,'invalid schedule');
    assert(a.lyingRate>=0&&a.lyingRate<=1,'invalid lying rate');
    assert(Object.entries(a.weights).every(([p,w])=>PURPOSES.includes(p)&&Number.isFinite(w)&&w>=0),'invalid purpose weight');
    s.stock[`${id}.cash`]=10; s.stock[`${id}.parts`]=0;
    s.beliefs[`${id}|shipment.cancelled`]={value:false,confidence:.7,source:'seed',learned_at:0};
    for(const other of Object.keys(cast)) if(other!==id) s.trust[`${id}|${other}`]=0;
  }
  return s;
}
function numeric(v:unknown):number {assert(typeof v==='number'&&Number.isFinite(v),'finite number required');return v;}
export function effects(state:State, changes:Effect[]):State {
  assert(changes.length<=16,'too many effects'); const s=clone(state);
  for(const e of changes){
    assert(e&&typeof e.target==='string'&&typeof e.value!=='object','invalid effect');
    if(e.op==='stock_delta'){assert(own(s.stock,e.target),'unknown stock');const n=numeric(e.value);assert(Math.abs(n)<=100&&Number.isInteger(n),'invalid stock delta');assert(s.stock[e.target]+n>=0,'insufficient stock');s.stock[e.target]+=n;}
    else if(e.op==='need_delta'){const [id,need]=e.target.split('|');assert(own(s.agents,id)&&needs.includes(need),'unknown need');s.agents[id].needs[need as Need]=clamp(s.agents[id].needs[need as Need]+numeric(e.value));}
    else if(e.op==='trust_delta'){assert(own(s.trust,e.target),'unknown relationship');s.trust[e.target]=clamp(s.trust[e.target]+numeric(e.value),-1,1);}
    else if(e.op==='fact_set'){assert(own(s.facts,e.target),'unknown fact');assert(typeof e.value===typeof s.facts[e.target],'fact type mismatch');s.facts[e.target]=e.value;}
    else if(e.op==='place_flag'){const [place,flag]=e.target.split('|');assert(own(s.places,place)&&['hot','cold','storm','smoke','rumor'].includes(flag),'unknown place flag');if(flag!=='rumor'){for(const old of ['hot','cold','storm','smoke'])delete s.flags[`${place}|${old}`];s.flags[`${place}|weather_until`]=s.minute+180;}s.flags[e.target]=e.value;}
    else if(e.op==='belief_set'){const [id,fact]=e.target.split('|');assert(own(s.agents,id)&&own(s.facts,fact),'unknown belief');assert(typeof e.value===typeof s.facts[fact],'belief type mismatch');assert(!!e.source,'belief source required');s.beliefs[e.target]={value:e.value,confidence:.7,source:e.source!,learned_at:s.minute};}
    else throw new Error('unknown effect operation');
  }
  return s;
}
export function checkCondition(s:State,c:Condition):boolean {
  const [type,...rest]=c.target.split(':');const key=rest.join(':');
  const table=type==='stock'?s.stock:type==='fact'?s.facts:null;
  assert(table&&own(table,key),'unknown goal target');const actual=table[key];
  assert(c.cmp==='eq'||c.cmp==='gte','unknown comparison');
  if(c.cmp==='gte'){assert(typeof actual==='number'&&typeof c.value==='number','numeric goal required');return actual>=c.value;}
  assert(typeof actual===typeof c.value,'goal type mismatch'); return actual===c.value;
}
export function applyEvent(state:State,e:Event):State {
  assert(e.id.length>0&&e.title.trim().length>0&&e.title.length<=200,'invalid event');
  assert(e.expires_at===undefined||e.expires_at>state.minute,'expired event');
  let s=effects(state,e.effects);
  if(e.quest){const q=clone(e.quest);assert(!own(s.quests,q.id)&&q.id===e.id,'duplicate quest');assert(q.status==='open'&&q.deadline>s.minute,'invalid quest deadline');assert(q.participants.length>=1&&q.participants.length<=8&&q.participants.every(id=>own(s.agents,id)),'unknown quest participant');assert(q.goal.length>0&&q.goal.length<=8&&q.steps.length<=8,'invalid goals');for(const c of [...q.goal,...q.steps.flat()])checkCondition(s,c);effects(s,q.reward);effects(s,q.penalty);s.quests[q.id]=q;}
  // A zero-effect event remains inspectable and talkable as a bounded public rumor.
  if(!e.effects.length&&!e.quest)s.flags['square|rumor']=e.title;
  s.version++;return s;
}
export function tick(state:State):{state:State;events:Event[]} {
  let s=clone(state);s.minute+=5; const events:Event[]=[];
  for(const place of Object.keys(s.places))if(Number(s.flags[`${place}|weather_until`]??Infinity)<=s.minute){for(const flag of ['hot','cold','storm','smoke','weather_until'])delete s.flags[`${place}|${flag}`];}
  for(const [id,a] of Object.entries(s.agents)){
    const hour=(s.minute/60)%24;const working=hour>=a.schedule.start&&hour<a.schedule.end;
    a.needs.rest=clamp(a.needs.rest+(working?.004:-.008));a.needs.social=clamp(a.needs.social+.003);if(s.flags['square|hot']||s.flags['square|cold'])a.needs.rest=clamp(a.needs.rest+.006);if(s.flags['square|storm']||s.flags['square|smoke'])a.needs.social=clamp(a.needs.social+.006);a.needs.money=clamp(1-s.stock[`${id}.cash`]/20);a.needs.purpose=clamp(a.needs.purpose+(Object.values(s.quests).some(q=>q.status==='open'&&q.participants.includes(id))?.015:-.002));
  }
  if(s.minute%60===0&&s.stock['diner.coffee']>0)s.stock['diner.coffee']=Math.max(0,s.stock['diner.coffee']-(s.flags['square|hot']||s.flags['square|cold']?2:1));
  // Poisson arrival probability for a five-minute interval, mean 1 delivery / 3h.
  if(draw(s)<1-Math.exp(-5/180)){
    const late=draw(s)<.25;const e:Event={id:`tick:${s.minute}`,source:'sim',kind:'delivery',title:late?'The parts shipment is cancelled.':'Parts arrived at the workshop.',effects:[{op:'fact_set',target:'shipment.cancelled',value:late},...(!late?[{op:'stock_delta' as const,target:'workshop.parts',value:3}]:[])]};s=applyEvent(s,e);events.push(e);
  }
  for(const q of Object.values(s.quests))if(q.status==='open'){
    const done=q.goal.every(c=>checkCondition(s,c));if(done||s.minute>=q.deadline){
      // Rewards/penalties are atomic. If resources were spent since injection,
      // leave the quest open for inspection instead of partially applying effects.
      s.quests[q.id].status=done?'succeeded':'failed';
      try{s=effects(s,done?q.reward:q.penalty);}catch(e){s.quests[q.id].effect_error=String(e);}
    }
  }
  s.version++;return {state:s,events};
}
export function legalResults(s:State,participants:string[],purpose:string):Result[]{
  assert(participants.length===2&&participants[0]!==participants[1]&&participants.every(p=>own(s.agents,p)),'invalid participants');
  if(purpose==='breeze')return [];const [a,b]=participants;const out:Result[]=[];
  const add=(id:string,title:string,es:Effect[])=>{try{effects(s,es);out.push({id,title,effects:es});}catch{/* unavailable */}};
  if(['ask_favor','persuade','quest_step'].includes(purpose)){
    add('lend_parts',`${s.agents[a].name} lends ${s.agents[b].name} one part`,[{op:'stock_delta',target:`${a}.parts`,value:-1},{op:'stock_delta',target:`${b}.parts`,value:1},{op:'trust_delta',target:`${b}|${a}`,value:.1}]);
    add('buy_part',`${s.agents[b].name} buys a workshop part for two cash`,[{op:'stock_delta',target:'workshop.parts',value:-1},{op:'stock_delta',target:`${b}.cash`,value:-2},{op:'stock_delta',target:`${b}.parts`,value:1}]);
  }
  if(['trade_news','deceive'].includes(purpose))for(const fact of Object.keys(s.facts)){
    const belief=s.beliefs[`${a}|${fact}`];if(!belief)continue;
    const value=purpose==='deceive'&&typeof belief.value==='boolean'?!belief.value:belief.value;
    add(`tell:${fact}`,`${s.agents[a].name} tells ${s.agents[b].name} ${fact} is ${value}`, [{op:'belief_set',target:`${b}|${fact}`,value,source:a}]);
  }
  if(purpose==='quest_step'&&Object.values(s.quests).some(q=>q.status==='open'&&participants.every(p=>q.participants.includes(p))))add('repair_generator',`${s.agents[b].name} repairs the generator using three parts`,[{op:'stock_delta',target:`${b}.parts`,value:-3},{op:'fact_set',target:'generator.fixed',value:true}]);
  add('agree',`${s.agents[b].name} agrees to help ${s.agents[a].name}`, [{op:'trust_delta',target:`${a}|${b}`,value:.05}]);return out;
}
export function choosePurpose(state:State,participants:string[]):{purpose:string;state:State}{
  const s=clone(state);const a=s.agents[participants[0]];const weights=PURPOSES.map(p=>(a.weights[p]??1)*(p==='breeze'?1+a.needs.social:p==='quest_step'?1+4*a.needs.purpose:p==='deceive'?a.lyingRate:1));
  let r=draw(s)*weights.reduce((a,b)=>a+b,0);for(let i=0;i<weights.length;i++){r-=weights[i];if(r<0)return {purpose:PURPOSES[i],state:s};}return {purpose:'breeze',state:s};
}
export function normalize(text:string):string{return text.toLowerCase().replace(/[^a-z0-9\s]/g,' ').replace(/\s+/g,' ').trim();}
export function echo(text:string,prior:string):boolean{
  const a=normalize(text),b=normalize(prior);if(!a||!b)return false;if(a===b)return true;
  const aa=new Set(a.split(' ')),bb=new Set(b.split(' '));if(Math.min(aa.size,bb.size)<5)return false;const intersection=[...aa].filter(x=>bb.has(x)).length;return intersection/(aa.size+bb.size-intersection)>=.8;
}
export function validateScene(value:unknown,s:State,participants:string[],purpose:string,legal:Result[]):Scene {
  assert(value&&typeof value==='object','scene object required');const scene=value as Scene;
  assert(Object.keys(scene).every(k=>['lines','results','why'].includes(k)),'unknown scene field');assert(Array.isArray(scene.lines)&&scene.lines.length>=3&&scene.lines.length<=6,'3-6 lines required');assert(Array.isArray(scene.results)&&scene.results.length<=(purpose==='breeze'?0:2),'too many results');assert(new Set(scene.results).size===scene.results.length&&scene.results.every(id=>legal.some(r=>r.id===id)),'illegal result');assert(typeof scene.why==='string'&&scene.why.trim().length>0&&scene.why.length<=300,'reason required');
  const names=participants.map(p=>s.agents[p].name);const entityWords=new Set([...Object.values(s.agents).flatMap(a=>a.name.split(' ')),...Object.values(s.places).map(p=>p.name),'Orion','I']);
  const starters=new Set(['The','That','This','It','We','You','Can','Could','Would','Do','Did','Is','Are','Yes','No','Thanks','Thank','Hello','Hi','My','Your','Our','Please','Let','If','But','And','So','Maybe','What','Where','When','Why','How','There']);
  scene.lines.forEach((l,i)=>{assert(Object.keys(l).every(k=>['speaker','text'].includes(k)),'unknown line field');assert(l.speaker===names[i%2],'speakers must alternate');assert(typeof l.text==='string'&&l.text.trim().length>0&&l.text.length<=400,'invalid line');if(i)assert(!echo(l.text,scene.lines[i-1].text),'partner echo');
    // Conservative proper-name gate. Structured IDs are checked independently.
    for(const word of l.text.match(/\b[A-Z][a-zA-Z]+\b/g)??[])assert(entityWords.has(word)||starters.has(word),'unknown named entity: '+word);
  });
  const selected=scene.results.flatMap(id=>legal.find(r=>r.id===id)!.effects);effects(s,selected);return scene;
}
export function resolveScene(s:State,scene:Scene,participants:string[],legal:Result[]):State{
  let next=effects(s,scene.results.flatMap(id=>legal.find(r=>r.id===id)!.effects));
  next=effects(next,participants.map(id=>({op:'need_delta',target:`${id}|social`,value:-.1} as Effect)));
  next=effects(next,[{op:'trust_delta',target:`${participants[0]}|${participants[1]}`,value:.01}]);next.version++;return next;
}
/** Multi-label binary Brier: results are not mutually exclusive (0–2 may occur). */
export function brier(probabilities:Record<string,number>,chosen:string[],ids:string[]):number{
  const keys=[...ids,'none'];assert(Object.keys(probabilities).length===keys.length&&keys.every(k=>own(probabilities,k)),'forecast keys mismatch');
  return keys.reduce((sum,k)=>{const p=numeric(probabilities[k]);assert(p>=0&&p<=1,'invalid probability');const y=k==='none'?Number(!chosen.length):Number(chosen.includes(k));return sum+(p-y)**2;},0)/keys.length;
}
export function frequencies(samples:string[][],ids:string[]):Record<string,number>{assert(samples.length>0,'no baseline samples');return Object.fromEntries([...ids,'none'].map(id=>[id,samples.filter(s=>id==='none'?!s.length:s.includes(id)).length/samples.length]));}

const fs=require('node:fs');
const assert=require('node:assert/strict');
const puppeteer=require('puppeteer');
(async()=>{
 const browser=await puppeteer.launch({headless:true,args:['--no-sandbox','--disable-setuid-sandbox']});
 try{
 const page=await browser.newPage();const sent=[];
 await page.setRequestInterception(true);
 page.on('request',req=>{
  if(req.url().endsWith('/api/aitown/simulation'))return req.respond({status:200,contentType:'application/json',body:JSON.stringify({world_id:'world',simulation:{enabled:true,state:{minute:15,stock:{'workshop.parts':3},quests:{},agents:{a:{name:'Mara Vale'},b:{name:'Nico Sable'}}}}})});
  if(req.url().endsWith('/api/aitown/events')){sent.push(JSON.parse(req.postData()));return req.respond({status:200,contentType:'application/json',body:JSON.stringify({accepted:true})});}
  if(req.url().endsWith('/api/aitown/status'))return req.respond({status:200,contentType:'application/json',body:'{}'});
  return req.respond({status:200,contentType:'text/html',body:fs.readFileSync(process.env.TOWN_UI_HTML,'utf8')});
 });
 await page.goto('http://town.test/');
 await page.addScriptTag({path:require('node:path').resolve(__dirname,'../static/js/aitown-panel.js')});
 await page.evaluate(()=>window.OrionAitownPanel.refresh());
 await page.waitForFunction(()=>document.querySelector('[name=participants]').options.length===2);
 await page.select('[name=kind]','quest');await page.select('[name=participants]','a','b');
 await page.type('[name=title]','Repair the generator');
 await page.evaluate(()=>document.querySelector('details').open=true);
 await page.click('#townEventForm button');
 await page.waitForFunction(()=>document.getElementById('townEventResult').textContent==='Added to the town.');
 assert.equal(sent.length,1);assert.equal(sent[0].quest.deadline,195);assert.deepEqual(sent[0].quest.participants,['a','b']);assert(sent[0].correlation_id);
 assert(!('source' in sent[0]));assert(!('world_id' in sent[0]));
 console.log('Town quest form: rendered controls -> request -> accepted response PASS');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

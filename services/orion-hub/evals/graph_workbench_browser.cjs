/* Real-browser eval. Usage: node ...cjs BASE_URL OUTPUT_DIR
 * Requires Hub's existing puppeteer dependency. Output stays local (private data).
 */
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const puppeteer = require(process.env.PUPPETEER_MODULE || "puppeteer");

(async () => {
  const [base, outputDir] = process.argv.slice(2);
  if (!base || !outputDir) throw new Error("Expected BASE_URL OUTPUT_DIR");
  const browser = await puppeteer.launch({headless:true, args:["--no-sandbox", "--enable-unsafe-swiftshader"]});
  const report = [];
  try {
    for (const source of ["worldview", "substrate", "crystallizations"]) {
      const page = await browser.newPage();
      await page.setViewport({width:1440,height:1000});
      const errors = [];
      page.on("pageerror", e => errors.push(e.message));
      await page.goto(`${base}/graph-workbench`, {waitUntil:"networkidle0"});
      assert(await page.$("#workbench-form"), "Rendered launcher missing");
      await page.select("#source", source);
      const targetPromise = browser.waitForTarget(t => t.opener() === page.target());
      await page.click("#open");
      const target = await targetPromise;
      const gephi = await target.page();
      await gephi.setViewport({width:1440,height:1000});
      gephi.on("pageerror", e => errors.push(e.message));
      gephi.on("console", msg => {if(msg.type() === "error") errors.push(msg.text());});
      gephi.on("response", res => {if(res.status() >= 400) console.log(`HTTP ${res.status()} ${res.url()}`);});
      await gephi.waitForFunction(() => document.title.includes("Gephi"), {timeout:30000});
      try { await gephi.waitForSelector("canvas", {timeout:30000}); }
      catch(error) {
        await gephi.screenshot({path:path.join(outputDir,`${source}-failure.png`)});
        console.log(JSON.stringify({source,errors,body:await gephi.$eval("body",e=>e.innerText)}));
        throw error;
      }
      await page.waitForFunction(() => !document.getElementById("open").disabled, {timeout:30000});
      const status = await page.$eval("#status", e => e.textContent);
      assert.match(status, /\d+ nodes · \d+ edges/);
      // The launcher preflights one live snapshot and Gephi fetches the same
      // reloadable URL again. Assert a real native import without pretending
      // those two independently read snapshots are byte-identical.
      await gephi.waitForFunction(() => {
        const text = document.body.innerText;
        return /Nodes\n[1-9]\d*/.test(text) && /Edges\n\d+/.test(text);
      });
      await gephi.click("::-p-text(Data)");
      await gephi.waitForSelector('table tbody tr');
      const properties = await gephi.$$eval("table thead th", cells => cells.map(c => c.innerText.trim()));
      assert(properties.some(p => p.includes("node_labels")));
      const expected = {worldview:"confidence",substrate:"activation",crystallizations:"summary"}[source];
      assert(properties.some(p => p.includes(expected)), `Missing imported property: ${expected}`);
      await gephi.click('table tbody input[type="checkbox"]');
      await gephi.waitForFunction(() => !document.body.innerText.includes("No selected node"));
      await gephi.screenshot({path:path.join(outputDir,`${source}-data.png`)});
      await gephi.click("::-p-text(Graph)");
      await gephi.waitForSelector("canvas");
      await gephi.locator("::-p-text(Layout)").click();
      await gephi.locator("::-p-text(Circular)").click();
      try { await gephi.waitForSelector('::-p-text(Apply)', {timeout:5000}); }
      catch(error) {
        await gephi.screenshot({path:path.join(outputDir,`${source}-layout-failure.png`)});
        throw error;
      }
      const coordinates = await gephi.$eval("body", e => e.innerText.match(/\nx\n([^\n]+)\ny\n([^\n]+)/)?.[0]);
      assert(coordinates, "Selected node has no displayed layout coordinates");
      await gephi.click('::-p-text(Apply)');
      await gephi.waitForFunction(before => {
        const current = document.body.innerText.match(/\nx\n([^\n]+)\ny\n([^\n]+)/);
        return current && current[0] !== before;
      }, {}, coordinates);
      await gephi.screenshot({path:path.join(outputDir,`${source}.png`)});
      const text = await gephi.$eval("body", e => e.innerText);
      const importedCounts = {
        nodes:Number(text.match(/Nodes\n([\d,]+)/)?.[1].replaceAll(",", "")),
        edges:Number(text.match(/Edges\n([\d,]+)/)?.[1].replaceAll(",", "")),
      };
      assert(importedCounts.nodes > 0, "Gephi imported no nodes");
      fs.writeFileSync(path.join(outputDir, `${source}-ui.txt`), text);
      report.push({source,status,importedCounts,canvas:true,properties,selectedNode:true,layoutChanged:true,errors});
      await gephi.close();
      await page.close();
    }
    // Known directed multigraph proves the real importer preserves edge IDs,
    // parallel relations, and reverse direction (no database fixture writes).
    const fixture = await browser.newPage();
    await fixture.setViewport({width:1440,height:1000});
    await fixture.goto(`${base}/gephi-lite/?file=${encodeURIComponent(`${base}/eval/parallel.gexf`)}`);
    await fixture.waitForFunction(() => document.body.innerText.includes("Nodes\n2") && document.body.innerText.includes("Edges\n3"));
    assert.match(await fixture.$eval("body",e=>e.innerText), /Directed graph/);
    await fixture.locator("::-p-text(Data)").click();
    await fixture.waitForSelector("table tbody tr");
    await fixture.locator('::-p-xpath(//button[normalize-space(.)="Edges"])').click();
    await fixture.waitForFunction(() => document.querySelector("table")?.innerText.includes("reverse"));
    const edges = await fixture.$$eval("table tbody tr", rows=>rows.map(row=>({
      id:row.querySelector('[data-field="id"]')?.innerText,
      source:row.querySelector('[data-field="sourceId"]')?.innerText,
      target:row.querySelector('[data-field="targetId"]')?.innerText,
    })));
    assert.deepEqual(edges.sort((a,b)=>a.id.localeCompare(b.id)), [
      {id:"first",source:"a",target:"b"}, {id:"reverse",source:"b",target:"a"}, {id:"second",source:"a",target:"b"},
    ]);
    report.push({source:"parallel-fixture",directedEdges:edges});
    await fixture.close();
    fs.writeFileSync(path.join(outputDir,"report.json"),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report,null,2));
    assert(report.every(r=>!r.errors?.length), "Browser errors occurred");
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});

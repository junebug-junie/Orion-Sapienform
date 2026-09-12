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
    const hub = await browser.newPage();
    await hub.goto(`${base}/eval/hub#graph-workbench`, {waitUntil:"domcontentloaded"});
    await hub.waitForFunction(() => {
      const panel = document.getElementById("graph-workbench");
      const frame = document.getElementById("graphWorkbenchPanelFrame");
      return panel && !panel.classList.contains("hidden") && frame?.getAttribute("src") === "/graph-workbench";
    }, {timeout:30000});
    assert.equal(await hub.evaluate(() => location.hash), "#graph-workbench");
    const pageCount = (await browser.pages()).length;
    await hub.click("#hubTabButton");
    await hub.waitForFunction(() => document.getElementById("graph-workbench")?.classList.contains("hidden"));
    await hub.click("#graphWorkbenchTabButton");
    await hub.waitForFunction(() => !document.getElementById("graph-workbench")?.classList.contains("hidden"));
    assert.equal((await browser.pages()).length, pageCount, "Graphs tab opened another browser page");
    assert.equal(await hub.evaluate(() => location.hash), "#graph-workbench");
    report.push({source:"hub-tab",deepLink:true,lazyFrame:true,restored:true,noPopup:true});
    await hub.close();

    for (const source of ["worldview", "substrate", "crystallizations"]) {
      const page = await browser.newPage();
      await page.setViewport({width:1440,height:1000});
      const errors = [];
      page.on("pageerror", e => errors.push(e.message));
      page.on("console", msg => {if(msg.type() === "error") errors.push(msg.text());});
      page.on("response", res => {if(res.status() >= 400) console.log(`HTTP ${res.status()} ${res.url()}`);});
      await page.goto(`${base}/graph-workbench`, {waitUntil:"networkidle0"});
      assert(await page.$("#graph-frame"), "Embedded Gephi frame missing");
      assert.equal(await page.$eval("#source", e => e.value), "crystallizations", "Memory should be the useful default");
      assert.equal(await page.$eval("details", e => e.open), false, "Advanced controls should start collapsed");
      assert.equal(await page.$("#open"), null, "A second open step must not return");
      await page.select("#source", source);
      await page.waitForFunction((selectedSource) => {
        const frame = document.getElementById("graph-frame");
        const focus = document.getElementById("seed");
        return frame?.dataset.source === selectedSource && frame?.src.includes("/gephi-lite/?file=blob")
          && frame.dataset.confirmedVersion
          && frame.dataset.confirmedVersion === frame.dataset.requestVersion
          && !focus?.disabled && focus?.options.length > 1;
      }, {timeout:30000}, source);
      const frameElement = await page.$("#graph-frame");
      const gephi = await frameElement.contentFrame();
      assert(gephi, "Embedded Gephi frame unavailable");
      await gephi.waitForFunction(() => document.title.includes("Gephi"), {timeout:30000});
      try { await gephi.waitForSelector("canvas", {timeout:30000}); }
      catch(error) {
        await page.screenshot({path:path.join(outputDir,`${source}-failure.png`)});
        console.log(JSON.stringify({source,errors,body:await gephi.$eval("body",e=>e.innerText)}));
        throw error;
      }
      const status = await page.$eval("#status", e => e.textContent);
      assert(status.length > 0, "Workbench status missing");
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
      await page.screenshot({path:path.join(outputDir,`${source}-data.png`)});
      await gephi.click("::-p-text(Graph)");
      await gephi.waitForSelector("canvas");
      await gephi.locator("::-p-text(Layout)").click();
      await gephi.locator("::-p-text(Circular)").click();
      try { await gephi.waitForSelector('::-p-text(Apply)', {timeout:5000}); }
      catch(error) {
        await page.screenshot({path:path.join(outputDir,`${source}-layout-failure.png`)});
        throw error;
      }
      const coordinates = await gephi.$eval("body", e => e.innerText.match(/\nx\n([^\n]+)\ny\n([^\n]+)/)?.[0]);
      assert(coordinates, "Selected node has no displayed layout coordinates");
      await gephi.click('::-p-text(Apply)');
      await gephi.waitForFunction(before => {
        const current = document.body.innerText.match(/\nx\n([^\n]+)\ny\n([^\n]+)/);
        return current && current[0] !== before;
      }, {}, coordinates);
      await page.screenshot({path:path.join(outputDir,`${source}.png`)});
      const text = await gephi.$eval("body", e => e.innerText);
      const importedCounts = {
        nodes:Number(text.match(/Nodes\n([\d,]+)/)?.[1].replaceAll(",", "")),
        edges:Number(text.match(/Edges\n([\d,]+)/)?.[1].replaceAll(",", "")),
      };
      assert(importedCounts.nodes > 0, "Gephi imported no nodes");
      fs.writeFileSync(path.join(outputDir, `${source}-ui.txt`), text);
      report.push({source,status,importedCounts,canvas:true,properties,selectedNode:true,layoutChanged:true,errors});
      if (source === "worldview") {
        const previous = await page.$eval("#graph-frame", e => ({src:e.src, version:e.dataset.requestVersion}));
        await page.click("#refresh");
        await page.waitForFunction((oldVersion) => {
          const frame = document.getElementById("graph-frame");
          return frame?.dataset.requestVersion !== oldVersion
            && frame.dataset.confirmedVersion === frame.dataset.requestVersion;
        }, {timeout:30000}, previous.version);
        const reloaded = await page.$eval("#graph-frame", e => ({
          src:e.src,
          requestVersion:e.dataset.requestVersion,
          confirmedDocumentVersion:e.dataset.confirmedDocumentVersion,
        }));
        assert.notEqual(reloaded.src, previous.src, "Refresh reused the old snapshot URL");
        assert.equal(reloaded.confirmedDocumentVersion, reloaded.requestVersion, "Same-count reload was confirmed against the old iframe document");
      }
      if (source === "crystallizations") {
        const focused = await page.$eval("#seed", e => e.options[1]?.value);
        assert(focused, "Focus dropdown did not contain a named memory");
        await page.select("#seed", focused);
        await page.waitForFunction((value) => document.getElementById("graph-frame")?.dataset.seed === value, {}, focused);
        await page.waitForFunction(() => document.getElementById("status")?.textContent.includes("Ready"), {timeout:30000});
        await page.type("#query", "zzzz-no-such-memory-zzzz");
        await page.waitForFunction(() => document.getElementById("focus-status")?.textContent.includes("0 matching"));
        assert.equal(await page.$eval("#seed", e => e.value), focused, "Filtering reset the active focus");
        assert.equal(await page.$eval("#graph-frame", e => e.dataset.seed), focused, "Filtering changed the displayed graph");
      }
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

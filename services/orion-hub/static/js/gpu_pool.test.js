const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const gp = require("./gpu_pool.js");

const CONFIG = {
  cards: { gpu0: { vram_gb: 32, lendable: true }, gpu1: { vram_gb: 32 }, gpu3: { vram_gb: 32 } },
  roles: {
    chat: { kind: "llm", cards: ["gpu0"], owner: ["chat"], port: 8011 },
    agent: { kind: "llm", cards: ["gpu1"], owner: ["agent"], port: 8015 },
    metacog: { kind: "llm", cards: ["gpu3"], owner: ["metacog", "fast"], port: 8012 },
    experiment: { kind: "llm", cards: ["gpu0", "gpu1", "gpu3"], owner: ["experiment"], port: 8099, operator_only: true, swap: {} },
  },
  classes: { chat: { roles: ["chat"] }, agent: { roles: ["agent", "chat"] }, metacog: { roles: ["metacog", "agent", "chat"] },
             fast: { roles: ["metacog"] }, experiment: { roles: ["experiment"] } },
};
const STATE = {
  cards: [{ card: "gpu0", lent: true, swapped_in: [] }],
  roles: [{ role: "metacog", status: "confirmed", slots: 4, ctx_per_slot: 4096, model_file: "Qwen_Qwen3-8B-Q5_K_M.gguf", profile_name: "p" },
          { role: "chat", status: "silent", slots: 1 }],
  leases: [{ lease_id: "a", role: "metacog", status: "granted" }, { lease_id: "b", role: "metacog", status: "queued" }],
};

test("cardModel places roles on their cards, spanning seats separately, with borrowers", () => {
  const m = gp.cardModel(CONFIG, STATE);
  const gpu3 = m.cards.find((c) => c.card === "gpu3");
  const metacog = gpu3.roles[0];
  assert.equal(metacog.modelFile, "Qwen_Qwen3-8B-Q5_K_M.gguf");
  assert.equal(metacog.busy, 1);                                  // queued leases hold no slot
  assert.deepEqual(metacog.borrowers, []);                         // fast is an owner, not a borrower
  assert.deepEqual(m.cards.find((c) => c.card === "gpu0").roles[0].borrowers, ["agent", "metacog"]);
  assert.equal(m.cards.find((c) => c.card === "gpu0").lent, true);
  assert.deepEqual(m.spanning.map((r) => r.name), ["experiment"]);
});

test("liveByRole counts slots in use now and grant waits from events", () => {
  const rows = gp.liveByRole(STATE, [
    { event: "granted", role: "metacog", waited_ms: 10 }, { event: "granted", role: "metacog", waited_ms: 30 },
    { event: "expired", role: "metacog" }, { event: "recalled", role: "chat" }]);
  const m = rows.find((r) => r.role === "metacog");
  assert.equal(m.busy, 1); assert.equal(m.grants, 2); assert.equal(m.failures, 1);
  assert.equal(m.wait_p50_ms, 20);
  assert.equal(rows.find((r) => r.role === "chat").recalls, 1);
});

test("liveByClass groups by class, holder and priority", () => {
  const rows = gp.liveByClass([
    { event: "granted", work_class: "metacog", holder: "gw", priority: "system", waited_ms: 5 },
    { event: "backlogged", work_class: "metacog", holder: "gw", priority: "system" },
    { event: "granted", work_class: "fast", holder: "gw", priority: "background", waited_ms: 1 }]);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].backlogged + rows[1].backlogged, 1);
});

test("walkerPath follows the lease's transitions with time between steps", () => {
  const p = gp.walkerPath([
    { event: "admit", status: "queued", at: "2026-09-24T12:00:00+00:00" },
    { event: "grant", from: "queued", status: "granted", role: "fast", at: "2026-09-24T12:00:02+00:00" },
    { event: "release_failed", from: "granted", status: "retry_wait", at: "2026-09-24T12:00:05+00:00" }]);
  assert.deepEqual(p.edges, [["queued", "granted"], ["granted", "retry_wait"]]);
  assert.equal(p.current, "retry_wait");
  assert.equal(p.steps[1].sincePrevMs, 2000);
});

test("walker EDGES mirror the lease graph's real transition table", () => {
  const src = fs.readFileSync(path.join(__dirname, "../../../../orion/gpu_pool/lease_graph.py"), "utf8");
  const table = src.slice(src.indexOf("_TABLE"), src.indexOf("}", src.indexOf("_TABLE")));
  const pairs = new Set();
  for (const m of table.matchAll(/\("(\w+)", "\w+"\): "(\w+)"/g)) if (m[1] !== m[2]) pairs.add(`${m[1]}>${m[2]}`);
  // retry_wait failures that exhaust attempts land on dead_letter / released (lease_graph.transition)
  ["granted>dead_letter", "recalling>dead_letter", "granted>released", "recalling>released"].forEach((p) => pairs.add(p));
  const drawn = new Set(gp.EDGES.map(([a, b]) => `${a}>${b}`));
  for (const p of pairs) assert.ok(drawn.has(p), `walker is missing edge ${p}`);
  for (const [a, b] of gp.EDGES) assert.ok(gp.NODES[a] && gp.NODES[b], `${a}>${b} has no node`);
});

test("seriesModel sums grants per bucket and keeps the worst p95", () => {
  const s = gp.seriesModel([{ t: "2026-09-24T12:00:00Z", grants: 2, wait_p95_ms: 10 },
                            { t: "2026-09-24T12:00:00Z", grants: 3, wait_p95_ms: 40 },
                            { t: "2026-09-24T12:01:00Z", grants: 1, wait_p95_ms: null }]);
  assert.equal(s.length, 2); assert.equal(s[0].grants, 5); assert.equal(s[0].p95, 40); assert.equal(s[1].p95, null);
});

test("fmtMs is readable at every scale", () => {
  assert.equal(gp.fmtMs(null), "–"); assert.equal(gp.fmtMs(250), "250 ms");
  assert.equal(gp.fmtMs(2500), "2.5 s"); assert.equal(gp.fmtMs(90000), "1.5 min");
});

test("fmtAt shows a clean UTC timestamp with milliseconds", () => {
  assert.equal(gp.fmtAt("2026-09-24T12:00:01+00:00"), "2026-09-24 12:00:01.000");
  assert.equal(gp.fmtAt(null), "");
});

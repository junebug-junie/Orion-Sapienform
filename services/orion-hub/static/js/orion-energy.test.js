const test = require("node:test");
const assert = require("node:assert/strict");
const tab = require("./orion-energy.js");

function bucket(over) {
  return Object.assign({
    label: "Wed Oct 21", complete: true, hours_elapsed: 24, kwh_by_source: {}, covered_hours_by_source: {},
    orion_kwh: null, house_kwh: null, house_hours: 0, overlap_hours: 0, orion_kwh_overlap: null,
    house_kwh_overlap: null, orion_share: null, orion_usd: null, orion_priced_kwh: 0, house_usd: null,
  }, over);
}

test("unknown renders as unknown, never 0", () => {
  assert.equal(tab.fmtKwh(null), "unknown");
  assert.equal(tab.fmtUsd(undefined), "unknown");
  assert.equal(tab.fmtPct(null), "unknown");
  assert.equal(tab.fmtKwh(0), "0.00 kWh");
  assert.equal(tab.fmtKwh(1234.4), "1,234 kWh");
  assert.equal(tab.fmtPct(0.5212), "52.1%");
});

test("color follows the source, not its position", () => {
  assert.equal(tab.colorFor("circe", ["circe"]), "var(--series-2)");
  assert.equal(tab.colorFor("circe", ["athena", "circe", "hecate", "cooling"]), "var(--series-2)");
  assert.equal(tab.colorFor("cooling", ["cooling"]), "var(--series-4)");
  assert.equal(tab.colorFor("atlas", ["athena", "atlas", "cooling"]), "var(--series-5)");
});

test("niceMax and label spacing", () => {
  assert.equal(tab.niceMax(0), 1);
  assert.equal(tab.niceMax(37), 50);
  assert.equal(tab.niceMax(396.7), 500);
  assert.equal(tab.labelEvery(744, 800), 66);
  assert.equal(tab.labelEvery(12, 800), 2);
});

test("tooltip names a missing source as no data and explains the share", () => {
  const b = bucket({
    kwh_by_source: { athena: 7.2 }, covered_hours_by_source: { athena: 23.8 }, orion_kwh: 7.2,
    house_kwh: 60, house_hours: 20, overlap_hours: 20, orion_share: 0.12, orion_usd: 1.1, orion_priced_kwh: 6,
  });
  const rows = tab.tooltipRows(b, ["athena", "cooling"]);
  assert.deepEqual(rows[0], { swatch: "var(--series-1)", name: "athena", value: "7.20 kWh", note: "99% of the time measured" });
  assert.equal(rows[1].name, "cabinet cooling");
  assert.equal(rows[1].value, "unknown");
  assert.equal(rows[1].note, "no data");
  const house = rows.find((r) => r.name === "House (RMP meter)");
  assert.equal(house.note, "20 of 24 h delivered");
  assert.equal(rows.find((r) => r.name === "Orion share").note, "over 20 h both meters have");
  assert.ok(tab.housePartial(b));
  assert.ok(!tab.housePartial(bucket({ house_kwh: 60, house_hours: 24 })));
});

test("window totals weight the share by kWh, not by averaging bucket shares", () => {
  const t = tab.windowTotals([
    bucket({ orion_kwh: 10, house_kwh: 20, overlap_hours: 24, orion_kwh_overlap: 10, house_kwh_overlap: 20, orion_share: 0.5 }),
    bucket({ orion_kwh: 10, house_kwh: 80, overlap_hours: 24, orion_kwh_overlap: 10, house_kwh_overlap: 80, orion_share: 0.125 }),
    bucket({}),
  ]);
  assert.equal(t.orion_kwh, 20);
  assert.equal(t.orion_kwh_overlap, 20);
  assert.equal(t.house_kwh_overlap, 100);
  assert.equal(t.overlap_hours, 48);
  assert.equal(t.orion_share, 0.2);
  assert.equal(t.orion_usd, null);
});

test("tooltip says how many partly measured hours the share left out", () => {
  const rows = tab.tooltipRows(bucket({ overlap_hours: 20, partial_hours: 2, orion_share: 0.5 }), []);
  assert.equal(rows.find((r) => r.name === "Orion share").note, "over 20 h both meters have; 2 partly measured h left out");
});

test("a failed load blanks the previous numbers instead of leaving them up", () => {
  const els = {};
  const doc = { getElementById: (id) => (els[id] = els[id] || { innerHTML: "old", textContent: "old" }) };
  tab.clearView(doc);
  assert.equal(els.oeKwhChart.innerHTML, "");
  assert.equal(els.oeTable.innerHTML, "");
  assert.equal(els.oeShare.textContent, "unknown");
  assert.equal(els.oeOrionUsd.textContent, "unknown");
});

test("an empty bucket blames the missing meter, not RMP", () => {
  const rows = tab.tooltipRows(bucket({ house_kwh: 323, house_hours: 146 }), ["athena"]);
  assert.equal(rows.find((r) => r.name === "Orion share").note, "no Orion data");
  assert.equal(rows.find((r) => r.name === "Orion cost").note, "no Orion data");
  const noHouse = tab.tooltipRows(bucket({ orion_kwh: 5, kwh_by_source: { athena: 5 } }), ["athena"]);
  assert.equal(noHouse.find((r) => r.name === "Orion share").note, "needs RMP data");
});

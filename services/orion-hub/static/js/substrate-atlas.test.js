const test = require("node:test");
const assert = require("node:assert/strict");
const { apiFetch } = require("./substrate-atlas.js");

function mockResponse({ ok, status, statusText, text, jsonThrows }) {
  let textCalls = 0;
  let jsonCalls = 0;
  return {
    ok,
    status: status ?? (ok ? 200 : 500),
    statusText: statusText ?? (ok ? "OK" : "Error"),
    text: async () => {
      textCalls += 1;
      return text ?? "";
    },
    json: async () => {
      jsonCalls += 1;
      if (jsonThrows) throw new Error("json should not be called");
      return JSON.parse(text ?? "{}");
    },
    get textCalls() {
      return textCalls;
    },
    get jsonCalls() {
      return jsonCalls;
    },
  };
}

test("apiFetch returns parsed JSON for OK responses", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    mockResponse({
      ok: true,
      text: JSON.stringify({ items: [{ trace_id: "t1" }] }),
    });
  try {
    const payload = await apiFetch("/api/substrate/atlas/traces?limit=1");
    assert.deepEqual(payload, { items: [{ trace_id: "t1" }] });
  } finally {
    global.fetch = originalFetch;
  }
});

test("apiFetch returns {} for OK empty body", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () => mockResponse({ ok: true, text: "" });
  try {
    const payload = await apiFetch("/api/substrate/atlas/traces?limit=1");
    assert.deepEqual(payload, {});
  } finally {
    global.fetch = originalFetch;
  }
});

test("apiFetch surfaces backend detail without double-reading the body", async () => {
  const originalFetch = global.fetch;
  let response;
  global.fetch = async () => {
    response = mockResponse({
      ok: false,
      status: 503,
      text: JSON.stringify({ detail: "grammar_atlas_disabled" }),
    });
    return response;
  };
  try {
    await assert.rejects(
      () => apiFetch("/api/substrate/atlas/traces?limit=50"),
      (err) => {
        assert.equal(err.message, "503: grammar_atlas_disabled");
        return true;
      },
    );
    assert.equal(response.textCalls, 1);
    assert.equal(response.jsonCalls, 0);
  } finally {
    global.fetch = originalFetch;
  }
});

test("apiFetch uses raw text when error body is not JSON", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    mockResponse({
      ok: false,
      status: 502,
      text: "Bad Gateway from proxy",
    });
  try {
    await assert.rejects(
      () => apiFetch("/api/substrate/atlas/traces?limit=50"),
      (err) => {
        assert.equal(err.message, "502: Bad Gateway from proxy");
        return true;
      },
    );
  } finally {
    global.fetch = originalFetch;
  }
});

test("apiFetch stringifies JSON error payloads without detail", async () => {
  const originalFetch = global.fetch;
  global.fetch = async () =>
    mockResponse({
      ok: false,
      status: 500,
      text: JSON.stringify({ error: "grammar_atlas_database_unconfigured" }),
    });
  try {
    await assert.rejects(
      () => apiFetch("/api/substrate/atlas/traces?limit=50"),
      (err) => {
        assert.equal(err.message, '500: {"error":"grammar_atlas_database_unconfigured"}');
        return true;
      },
    );
  } finally {
    global.fetch = originalFetch;
  }
});

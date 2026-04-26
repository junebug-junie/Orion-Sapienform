const test = require('node:test');
const assert = require('node:assert/strict');
const thoughtProcess = require('./thought-process.js');

test('selectThoughtProcess prefers reasoning_trace.content', () => {
  const out = thoughtProcess.selectThoughtProcess({
    reasoning_trace: { content: 'trace content', text: 'trace text' },
    reasoning_content: 'fallback',
  });
  assert.equal(out.text, 'trace content');
  assert.equal(out.source, 'reasoning_trace');
});

test('selectThoughtProcess falls back to reasoning_trace.text', () => {
  const out = thoughtProcess.selectThoughtProcess({
    reasoning_trace: { text: 'trace text only' },
    reasoning_content: 'fallback',
  });
  assert.equal(out.text, 'trace text only');
  assert.equal(out.source, 'reasoning_trace');
});

test('selectThoughtProcess uses reasoning_content when trace missing', () => {
  const out = thoughtProcess.selectThoughtProcess({
    reasoning_content: 'content path',
  });
  assert.equal(out.text, 'content path');
  assert.equal(out.source, 'reasoning_content');
});

test('selectThoughtProcess uses reasoning metacog trace', () => {
  const out = thoughtProcess.selectThoughtProcess({
    metacog_traces: [
      { trace_role: 'stance', content: 'not selected' },
      { trace_role: 'reasoning', content: 'metacog reasoning' },
    ],
  });
  assert.equal(out.text, 'metacog reasoning');
  assert.equal(out.source, 'metacog');
});

test('selectThoughtProcess uses provider raw reasoning_content', () => {
  const out = thoughtProcess.selectThoughtProcess({
    raw: {
      metadata: { reasoning_content: 'provider reasoning', model: 'm1', provider: 'p1' },
    },
  });
  assert.equal(out.text, 'provider reasoning');
  assert.equal(out.source, 'provider');
  assert.equal(out.metadata.model, 'm1');
  assert.equal(out.metadata.provider, 'p1');
});

test('selectThoughtProcess returns null text when absent', () => {
  const out = thoughtProcess.selectThoughtProcess({});
  assert.equal(out.text, null);
  assert.equal(out.source, null);
});

test('selectThoughtProcess ignores empty strings', () => {
  const out = thoughtProcess.selectThoughtProcess({
    reasoning_trace: { content: '   ' },
    reasoning_content: '',
    metacog_traces: [{ trace_role: 'reasoning', content: '  ' }],
  });
  assert.equal(out.text, null);
  assert.equal(out.source, null);
});

test('selectThoughtProcess supports older message shape', () => {
  const out = thoughtProcess.selectThoughtProcess({
    reasoningTrace: { content: 'legacy trace' },
    routing_debug: { mode: 'brain' },
  });
  assert.equal(out.text, 'legacy trace');
  assert.equal(out.source, 'reasoning_trace');
  assert.equal(out.metadata.mode, 'brain');
});

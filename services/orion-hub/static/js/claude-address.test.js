const test = require('node:test');
const assert = require('node:assert/strict');
const { parseClaudeAddress } = require('./claude-address.js');

test('a leading @claude goes to Claude with the prefix stripped', () => {
  assert.deepEqual(parseClaudeAddress('@claude what do you make of this?'), {
    toClaude: true,
    prompt: 'what do you make of this?',
  });
});

test('the address is case-insensitive and tolerates surrounding whitespace', () => {
  assert.deepEqual(parseClaudeAddress('  @Claude hi  '), { toClaude: true, prompt: 'hi' });
  assert.deepEqual(parseClaudeAddress('@CLAUDE hi'), { toClaude: true, prompt: 'hi' });
});

test('one separator after the address is dropped', () => {
  assert.equal(parseClaudeAddress('@claude: why?').prompt, 'why?');
  assert.equal(parseClaudeAddress('@claude, why?').prompt, 'why?');
  assert.equal(parseClaudeAddress('@claude\nline two').prompt, 'line two');
});

test('an empty remainder still routes to Claude with an empty prompt', () => {
  // app.js then falls back to the last room message, same as the button.
  assert.deepEqual(parseClaudeAddress('@claude'), { toClaude: true, prompt: '' });
  assert.deepEqual(parseClaudeAddress('  @claude :  '), { toClaude: true, prompt: '' });
});

test('@claude mid-sentence is ordinary text for Orion', () => {
  assert.deepEqual(parseClaudeAddress('ask @claude about this'), {
    toClaude: false,
    prompt: 'ask @claude about this',
  });
});

test('@claudette-style names are not an address', () => {
  for (const text of ['@claudette hi', '@claude_bot hi', '@claude2 hi', 'claude hi', '@ claude hi']) {
    assert.equal(parseClaudeAddress(text).toClaude, false, text);
  }
});

test('empty and non-string input go nowhere special', () => {
  assert.deepEqual(parseClaudeAddress(''), { toClaude: false, prompt: '' });
  assert.deepEqual(parseClaudeAddress(null), { toClaude: false, prompt: '' });
  assert.deepEqual(parseClaudeAddress(undefined), { toClaude: false, prompt: '' });
});

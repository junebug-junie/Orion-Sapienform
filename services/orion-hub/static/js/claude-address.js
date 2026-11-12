(function (global) {
  // "@claude ..." typed at the start of the chat box goes to Claude, never to
  // Orion. Pure so the routing decision is unit-testable without a DOM; app.js
  // calls it from the composer send path (sendTextMessage and the expanded
  // composer's send) and for the amber "this goes to Claude" cue.
  //
  // Only a LEADING @claude counts. "@claude" mid-sentence is ordinary text for
  // Orion, and "@claudette" / "@claude_x" are not an address (\b needs a
  // non-word character after "claude").
  const ADDRESS = /^@claude\b/i;
  // One separator right after the address is part of the address, not the
  // message: "@claude: why?" and "@claude, why?" both send "why?".
  const LEADING_SEPARATOR = /^\s*[,:;]?\s*/;

  function parseClaudeAddress(text) {
    const value = String(text == null ? '' : text).trim();
    if (!ADDRESS.test(value)) return { toClaude: false, prompt: value };
    const rest = value.replace(ADDRESS, '').replace(LEADING_SEPARATOR, '').trim();
    return { toClaude: true, prompt: rest };
  }

  const api = { parseClaudeAddress: parseClaudeAddress };

  global.OrionClaudeAddress = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

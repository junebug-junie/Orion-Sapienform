(function (global) {
  // "@claude ..." typed at the start of the chat box goes to Claude, never to
  // Orion. Pure so the routing decision is unit-testable without a DOM; app.js
  // calls it from the composer send path (sendTextMessage and the expanded
  // composer's send) and for the amber "this goes to Claude" cue.
  //
  // Only a LEADING @claude counts. "@claude" mid-sentence is ordinary text for
  // Orion. The address must end the text or be followed by whitespace or one
  // of , : ; -- so "@claudette", "@claude_x", "@claude-bot", "@claude.ai" and
  // "@claude's take" are ordinary text too (a bare \b would route the last
  // three and send Claude "'s take"). app.js carries a copy of this pattern as
  // a fail-closed fallback for when this file does not load; keep them equal
  // (tests/test_claude_address_fallback_parity.py).
  const ADDRESS = /^@claude(?=$|[\s,:;])/i;
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

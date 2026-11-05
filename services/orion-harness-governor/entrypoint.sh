#!/bin/sh
set -e

# GITHUB_PAT in ~/.fcc/.env is Orion's GitHub MCP token and sandbox push token.
# It comes ONLY from ORION_GITHUB_TOKEN (Orion's own GitHub user). The old
# fallback that copied the operator's gh login out of ~/.config/gh/hosts.yml is
# gone on purpose, and any stale GITHUB_PAT is removed first -- the previous
# "only if missing" guard left the operator's token in the file forever.
mkdir -p ~/.fcc
touch ~/.fcc/.env
if grep -q "^GITHUB_PAT=" ~/.fcc/.env; then
  grep -v "^GITHUB_PAT=" ~/.fcc/.env > ~/.fcc/.env.tmp || true
  cat ~/.fcc/.env.tmp > ~/.fcc/.env   # same inode: host bind mount keeps working
  rm -f ~/.fcc/.env.tmp
fi
if [ -n "$ORION_GITHUB_TOKEN" ]; then
  echo "GITHUB_PAT=$ORION_GITHUB_TOKEN" >> ~/.fcc/.env
fi

# Inject FIRECRAWL_API_KEY from env var if set and missing from file
if [ -n "$FIRECRAWL_API_KEY" ] && ! grep -q "^FIRECRAWL_API_KEY=" ~/.fcc/.env 2>/dev/null; then
  echo "FIRECRAWL_API_KEY=$FIRECRAWL_API_KEY" >> ~/.fcc/.env
fi

exec "$@"

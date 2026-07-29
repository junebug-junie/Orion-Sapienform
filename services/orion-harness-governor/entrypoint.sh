#!/bin/sh
set -e

# Inject MCP credentials into ~/.fcc/.env from gh credentials or env vars
# This handles FCC server regenerating the file and wiping manual entries

if [ -f ~/.config/gh/hosts.yml ]; then
  if ! grep -q "^GITHUB_PAT=" ~/.fcc/.env 2>/dev/null; then
    PAT=$(python3 << 'PYEOF'
import yaml
from pathlib import Path
try:
    hosts = Path.home() / ".config/gh/hosts.yml"
    if hosts.exists():
        config = yaml.safe_load(hosts.read_text()) or {}
        gh_com = config.get("github.com", {})
        if isinstance(gh_com, dict):
            token = gh_com.get("oauth_token")
            if token:
                print(token)
except:
    pass
PYEOF
)
    if [ -n "$PAT" ]; then
      if [ -f ~/.fcc/.env ]; then
        echo "GITHUB_PAT=$PAT" >> ~/.fcc/.env
      else
        mkdir -p ~/.fcc && echo "GITHUB_PAT=$PAT" > ~/.fcc/.env
      fi
    fi
  fi
fi

# Inject FIRECRAWL_API_KEY from env var if set and missing from file
if [ -n "$FIRECRAWL_API_KEY" ] && ! grep -q "^FIRECRAWL_API_KEY=" ~/.fcc/.env 2>/dev/null; then
  echo "FIRECRAWL_API_KEY=$FIRECRAWL_API_KEY" >> ~/.fcc/.env
fi

exec "$@"

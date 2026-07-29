#!/bin/sh
set -e

# Inject GITHUB_PAT from gh auth token into ~/.fcc/.env if missing
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

exec "$@"

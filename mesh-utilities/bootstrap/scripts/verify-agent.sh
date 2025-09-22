#!/usr/bin/env bash
set -euo pipefail
echo "🔍 SSH_AUTH_SOCK: ${SSH_AUTH_SOCK:-<empty>}"
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  echo "❌ No SSH agent detected. Connect with: ssh -A <user>@<node>"
  exit 1
fi
echo "🔐 Testing GitHub auth via forwarded agent..."
ssh -o IdentitiesOnly=yes -T git@github.com || true
REPO_URL="$(tr -d '\n' < "$(dirname "$0")/../config/docker-compose.repo" || true)"
if [[ -z "$REPO_URL" ]]; then
  echo "❌ config/docker-compose.repo is empty."
  exit 1
fi
echo "🔗 Listing refs for $REPO_URL ..."
GIT_SSH_COMMAND="ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" git ls-remote "$REPO_URL" && echo "✅ Agent forwarding works."

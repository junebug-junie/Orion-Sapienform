# services/orion-self-study-enrichment/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Required for the root FCC claude subprocess to start under full-auto-approve
# permissions instead of crashing on startup (even with tools disabled via
# --tools "" -- the startup gate fires regardless) -- see
# orion/fcc/claude_spawn.py::claude_permission_argv() for the canonical
# explanation (don't duplicate it here; that's the one place to update if
# Claude Code's own gate behavior changes).
ENV IS_SANDBOX=1

# - git: evidence.py reads graphify-out/graph.json + README.md from the
#   read-only repo mount (SELF_STUDY_ENRICHMENT_REPO_PATH) -- git itself
#   isn't shelled out to by this service, but the mount is a real .git
#   checkout so `git config --system --add safe.directory` still applies
#   the same way orion-cocreation-signals' Dockerfile documents (ownership
#   mismatch between host-owned mount and root-run container).
# - nodejs/npm: the Claude Code CLI (`claude`) this service spawns via
#   claude_runner.py is a Node package (@anthropic-ai/claude-code) -- no
#   existing Dockerfile in this repo installs it (FCC's own claude spawns
#   run on the host, not in a container), so this is a new install path.
#
# Node 22, not 20: confirmed live 2026-08-21 (same bug as
# orion-room-companion's identical fix, 2026-08-18) -- `npm install -g
# @anthropic-ai/claude-code` with no version pin does NOT install the
# latest published release when the local Node doesn't satisfy its
# `engines.node` constraint -- it silently resolves to the newest version
# that DOES, with no warning. Under Node 20 this installed 2.1.197 instead
# of the current release (which requires node >=22), and the stale CLI
# then failed `claude -p` auth with a 401 against a token the host's
# up-to-date CLI accepted fine -- looks exactly like a bad credential until
# versions are compared directly. `docker exec <container> claude --version`
# vs `claude --version` on the host is the fast way to notice this.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory '*'

WORKDIR /app

COPY services/orion-self-study-enrichment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-self-study-enrichment/app /app/app
COPY orion /app/orion

CMD ["python", "-m", "app.main"]

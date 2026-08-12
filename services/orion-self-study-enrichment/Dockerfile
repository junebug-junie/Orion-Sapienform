# services/orion-self-study-enrichment/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
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

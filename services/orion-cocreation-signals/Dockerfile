# services/orion-cocreation-signals/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# - git: git_delta.py/graph_delta.py shell out to `git` against the
#   read-only repo mount (COCREATION_SIGNALS_REPO_PATH).
# - gh (GitHub CLI): pr_lifecycle.py shells out to `gh pr list`. Installed
#   from GitHub's own apt repo (not in Debian's default repos on this base
#   image). Auth is via GH_TOKEN/GITHUB_TOKEN env var at runtime (gh honors
#   both natively) -- no `gh auth login` step, no mounted credential file.
# - curl, ca-certificates: needed to fetch/verify GitHub's apt signing key.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    mkdir -p -m 755 /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# Confirmed live via a real smoke test (docker compose up, real repo bind
# mount): git refuses to operate on a mounted directory it doesn't own
# ("detected dubious ownership in repository") -- the host repo is owned by
# the host user, this container runs as root. Wildcard, not the specific
# COCREATION_SIGNALS_REPO_PATH default (/repo), since that path is operator-
# configurable and this trust decision needs to hold regardless of where the
# mount lands. Safe here specifically because the mount is read-only and its
# content is this repo's own trusted checkout -- the protection this git
# safety check exists for (a malicious repo's config/hooks running with your
# privileges) doesn't apply to a read-only mount of a repo you already trust.
RUN git config --system --add safe.directory '*'

WORKDIR /app

COPY services/orion-cocreation-signals/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-cocreation-signals /app
COPY orion /app/orion

CMD ["python", "-m", "app.main"]

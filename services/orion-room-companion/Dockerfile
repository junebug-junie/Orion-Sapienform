# services/orion-room-companion/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Required for the root claude subprocess to start under full-auto-approve
# permissions instead of crashing on startup (the gate fires even with tools
# disabled via --tools "") -- see orion/fcc/claude_spawn.py::claude_permission_argv()
# for the canonical explanation. Don't duplicate it here; that's the one place
# to update if Claude Code's own gate behavior changes.
ENV IS_SANDBOX=1

# nodejs/npm: the Claude Code CLI (`claude`) this service spawns is a Node
# package. Installed into the image rather than bind-mounted from the host
# (orion-hub mounts the host binary) so this container's dependency surface is
# self-contained -- it deliberately has no repo mount to borrow from.
#
# No git, no gh, no ssh client: this service reads one credential file and
# talks to the bus. Every tool absent here is one a compromised room turn
# cannot reach for.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    apt-get purge -y --auto-remove gnupg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/orion-room-companion/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-room-companion/app /app/app
# scripts/ ships in the image on purpose: README documents the live smoke as
# `docker compose exec room-companion python -m scripts.smoke_room_turn`, and
# an acceptance surface that has to be hand-copied into a running container is
# not reproducible from the committed artifact.
COPY services/orion-room-companion/scripts /app/scripts
COPY orion /app/orion

CMD ["python", "-m", "app.main"]

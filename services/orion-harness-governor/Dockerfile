FROM python:3.12-slim

WORKDIR /app

# MCP toolchain: docker CLI (GitHub MCP), node/npx (Firecrawl MCP), git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    git \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && printf "Types: deb\nURIs: https://download.docker.com/linux/debian\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n" \
         "$(. /etc/os-release && echo "${VERSION_CODENAME}")" \
         "$(dpkg --print-architecture)" \
         > /etc/apt/sources.list.d/docker.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         docker-ce-cli \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory /mnt/scripts/Orion-Sapienform \
    && git config --global --add safe.directory '*'

# Self-indexing MCP toolchain: GitNexus (code graph) + Context Mode (context
# virtualization). Pinned exact versions — both are third-party executables
# inside the FCC tool loop; upgrade only after reviewing release diffs.
# context-mode requires node >=22.5 (nodesource 22.x currently ships >=22.5).
RUN node -e "const [maj,min]=process.versions.node.split('.').map(Number); if (maj<22||(maj===22&&min<5)) { throw new Error('need node >=22.5, got '+process.versions.node); }" \
    && npm install -g gitnexus@1.6.9 context-mode@1.0.169 \
    && gitnexus --version \
    && npm ls -g --depth=0 context-mode

# Bake the LadybugDB FTS extension into the image (lands in /root/.lbdb).
# GitNexus query paths are offline-first (load-only): without a pre-installed
# extension the MCP degrades to "FTS indexes missing" and keyword search over
# an FTS-built index returns nothing. --verify-only proves a fresh process can
# LOAD it so a HOME mismatch fails the build instead of silently degrading.
RUN node /usr/lib/node_modules/gitnexus/scripts/install-duckdb-extension.mjs fts 4294967296 \
    && node /usr/lib/node_modules/gitnexus/scripts/install-duckdb-extension.mjs fts 4294967296 --verify-only

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-harness-governor/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-harness-governor/app ./app
COPY orion ./orion

# Orion AI Town MCP server (python3 -m orion_aitown_mcp when HARNESS_AITOWN_ENABLED)
COPY services/orion-ai-town/mcp ./services/orion-ai-town/mcp
RUN pip3 install --no-cache-dir ./services/orion-ai-town/mcp

ENV PYTHONPATH=/app:/app/services/orion-ai-town/mcp

EXPOSE 7156

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7156"]

FROM python:3.12-slim

# LadybugDB FTS extension version/platform -- the on-disk path DuckDB LOADs
# from is decoupled from the gitnexus/@ladybugdb npm package's own semver
# (confirmed live: gitnexus@1.6.9 resolves @ladybugdb/core@0.18.3 but still
# expects the extension at the 0.18.1 path). Defaults match
# scripts/seed_lbdb_fts_extension_cache.sh's own LBDB_EXT_VERSION/
# LBDB_EXT_PLATFORM defaults and docker-compose.yml's build.args -- bump the
# env var in services/orion-harness-governor/.env, not this file, when the
# extension version changes.
ARG LBDB_EXT_VERSION=0.18.1
ARG LBDB_EXT_PLATFORM=linux_amd64

WORKDIR /app

# MCP toolchain: docker CLI (GitHub MCP), node/npx (Firecrawl MCP), git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    git \
    openssh-client \
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
    && git config --global --add safe.directory '*' \
    && git config --global user.name "Orion FCC" \
    && git config --global user.email "orion-fcc@orion-sapienform.local"

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
#
# Copied from a host-local cache (see scripts/seed_lbdb_fts_extension_cache.sh
# and the `lbdb_ext` additional_contexts entry in docker-compose.yml) instead
# of running install-duckdb-extension.mjs's own network install step -- that
# hits extension.ladybugdb.com directly, which has had real sustained outages
# (Cloudflare 522, observed 2026-08-12) that blocked this build entirely.
# Exact target path/version confirmed live by running the installer in an
# unmodified python:3.12-slim container and inspecting where it wrote.
COPY --from=lbdb_ext libfts.lbug_extension /root/.lbdb/extension/${LBDB_EXT_VERSION}/${LBDB_EXT_PLATFORM}/fts/libfts.lbug_extension
RUN node /usr/lib/node_modules/gitnexus/scripts/install-duckdb-extension.mjs fts 4294967296 --verify-only

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-harness-governor/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-harness-governor/app ./app
COPY services/orion-harness-governor/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
COPY orion ./orion

# Orion AI Town MCP server (python3 -m orion_aitown_mcp when HARNESS_AITOWN_ENABLED)
COPY services/orion-ai-town/mcp ./services/orion-ai-town/mcp
RUN pip3 install --no-cache-dir ./services/orion-ai-town/mcp

ENV PYTHONPATH=/app:/app/services/orion-ai-town/mcp

EXPOSE 7156

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7156"]

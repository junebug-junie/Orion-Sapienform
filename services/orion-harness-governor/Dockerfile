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
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-harness-governor/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-harness-governor/app ./app
COPY orion ./orion

# Orion AI Town MCP server (python3 -m orion_aitown_mcp when HARNESS_AITOWN_ENABLED)
COPY services/orion-ai-town/mcp ./services/orion-ai-town/mcp
RUN pip3 install --no-cache-dir ./services/orion-ai-town/mcp

ENV PYTHONPATH=/app/services/orion-ai-town/mcp

EXPOSE 7156

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7156"]

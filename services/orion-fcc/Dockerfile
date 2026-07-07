FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY services/orion-fcc/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY services/orion-fcc/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8082

ENTRYPOINT ["entrypoint.sh"]
CMD ["fcc-server"]

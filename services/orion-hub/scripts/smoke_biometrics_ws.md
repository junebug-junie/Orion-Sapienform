# Hub Biometrics WS Smoke Test

1) Start biometrics + hub (example):
```bash
docker compose -f services/orion-biometrics/docker-compose.yml --env-file .env up -d orion-biometrics
docker compose -f services/orion-hub/docker-compose.yml --env-file services/orion-hub/.env_example up -d hub-app
```

2) Open the hub UI:
```
http://localhost:8080
```

3) Verify the Biometrics panel updates within ~60s (status changes to LIVE/STALE, values populate).

Optional: watch biometrics bus traffic:
```bash
redis-cli -u "${ORION_BUS_URL}" psubscribe "orion:biometrics:*"
```

# services/orion-docker-door/Dockerfile -- stdlib only, no requirements.
FROM python:3.12-slim
WORKDIR /app
COPY services/orion-docker-door/app ./app
EXPOSE 2375
CMD ["python3", "-m", "app.door"]

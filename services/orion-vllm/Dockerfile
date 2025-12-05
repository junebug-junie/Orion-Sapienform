# services/orion-vllm/Dockerfile

FROM vllm/vllm-openai:v0.12.0

WORKDIR /app

# (Optional) if you have extra deps:
COPY services/orion-vllm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-vllm/app ./app
COPY config/llm_profiles.yaml ./config/llm_profiles.yaml
COPY orion ./orion

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "app.main"]

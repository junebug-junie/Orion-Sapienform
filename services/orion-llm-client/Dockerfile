FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .
CMD ["python","-c","import orion_llm_client; print('orion-llm-client ready')"]

FROM python:3.12-slim

WORKDIR /app

# Upgrade pip early
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
COPY services/orion-chat-memory/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in shared Orion libraries
COPY orion /app/orion

# Copy the service code
COPY services/orion-chat-memory/app /app/app

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

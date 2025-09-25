FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY services/orion-gdb-client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY services/orion-gdb-client/main.py .

CMD ["python", "main.py"]

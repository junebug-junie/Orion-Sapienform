#!/bin/sh
# This script ensures that DNS resolution is working before starting the main application.
# This prevents crashes when the container starts faster than the Docker network initializes.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Waiting for DNS resolution..."

# Loop until 'getent hosts' can successfully resolve huggingface.co.
# 'getent' is a standard Linux utility for querying databases, including DNS.
while ! getent hosts huggingface.co > /dev/null; do
  echo "DNS not ready, retrying in 2 seconds..."
  sleep 2
done

echo "DNS is ready. Starting application."

# Execute the command passed into this script.
# In our case, this will be the 'CMD' from the Dockerfile:
# ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8080"]
exec "$@"

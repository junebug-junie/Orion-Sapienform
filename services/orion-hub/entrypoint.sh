#!/bin/sh
# This script ensures that internal Docker DNS resolution is working before
# starting the main application. This prevents crashes when the container
# starts faster than the network initializes.

set -e

echo "Waiting for internal DNS..."

# Loop until 'getent hosts' can successfully resolve the bus service.
# This confirms the Docker network is ready for inter-service communication.
# The ${PROJECT} variable is passed in from the docker-compose.yml environment.
while ! getent hosts ${PROJECT}-bus-core > /dev/null; do
  echo "Internal DNS not ready for ${PROJECT}-bus-core, retrying in 2 seconds..."
  sleep 2
done

echo "Internal DNS is ready. Starting application."

# Execute the command passed into this script (the CMD from the Dockerfile).
exec "$@"


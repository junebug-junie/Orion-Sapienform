# Use a Python 3.11 base image for broad compatibility
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install git, which is required by sentence-transformers for some models
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container
# This includes the app/, scripts/, and config/ directories
COPY . .

# Set the PYTHONPATH so Python can find modules in the /app directory
ENV PYTHONPATH="/app:${PYTHONPATH}"

# This container is designed for executing scripts, not running a server.
# The default command is an interactive bash shell.
CMD ["bash"]

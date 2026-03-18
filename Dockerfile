FROM python:3.11-slim

WORKDIR /app

# Install git (needed for pip to fetch config-morpher from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/

# Install CPU-only PyTorch first (avoid pulling CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Default command
CMD ["guardrail-api", "--host", "0.0.0.0", "--port", "8000"]

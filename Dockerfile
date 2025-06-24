# Use specific Python version (3.10.17) for consistency
ARG PYTHON_VERSION=3.10.17
FROM python:${PYTHON_VERSION}-slim

# Label for base Python version
LABEL org.opencontainers.image.base.name="python:${PYTHON_VERSION}-slim"

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# Install project dependencies directly from pyproject.toml (PEP-517)
# -----------------------------------------------------------------

# Copy project metadata first to leverage Docker layer cache
COPY pyproject.toml README.md ./

# Install runtime dependencies defined in pyproject.toml (production only)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/
COPY main.py .

# Create necessary directories
RUN mkdir -p models data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Run the application
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--host", "0.0.0.0", "--port", "80"] 
# Ultra-minimal Docker build for Railway deployment
FROM python:3.11-alpine as builder

# Install minimal build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    && pip install --no-cache-dir --upgrade pip

# Install minimal Python dependencies
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Ultra-minimal production stage
FROM python:3.11-alpine

# Minimal environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install only curl for health checks
RUN apk add --no-cache curl

# Copy only essential Python packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy only essential application files
COPY app/ ./app/
COPY requirements-minimal.txt ./

# Create minimal directories
RUN mkdir -p models logs

# Expose port
EXPOSE 8000

# Minimal health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=2 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Run with minimal workers for Railway constraints
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
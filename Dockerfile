FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
# COPY src/ ./src/

# Create temp directory for file uploads
RUN mkdir temp

ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose port
EXPOSE 8001

# Run the FastAPI application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

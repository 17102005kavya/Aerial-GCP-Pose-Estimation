FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and torchvision to keep image size small (crucial for quick serverless spins)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart gunicorn requests

# Copy backend codebase
COPY . .

# Expose port (Hugging Face Spaces uses 7860, others inject PORT)
EXPOSE 7860

# Run with Gunicorn using Uvicorn workers, binding dynamically
CMD ["sh", "-c", "gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:${PORT:-7860}"]

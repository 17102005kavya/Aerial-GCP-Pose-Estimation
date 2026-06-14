# Production Deployment Guide

This guide details the steps to deploy the AeroPoint AI platform into production.

---

## 1. Frontend Deployment (Next.js 15)

The Next.js frontend can be deployed serverless (Vercel) or containerized (Docker).

### Option A: Vercel (Recommended)
Vercel is the native platform for Next.js, offering global CDN caching, automatic serverless scaling, and SSL out of the box.

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Run deployment from the `frontend/` directory:
   ```bash
   vercel
   ```
3. Set the environment variable:
   * Key: `NEXT_PUBLIC_API_URL`
   * Value: The public URL of your deployed FastAPI backend (e.g., `https://api.aeropoint.example.com`).

---

### Option B: Containerized (Docker)
For self-hosting on AWS EC2, DigitalOcean, or Azure VMs.

1. Next.js is configured for standalone output. Create a `Dockerfile` in the `frontend/` directory:
   ```dockerfile
   FROM node:18-alpine AS base
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   RUN npm run build

   FROM node:18-alpine AS runner
   WORKDIR /app
   ENV NODE_ENV=production
   COPY --from=base /app/public ./public
   COPY --from=base /app/.next/standalone ./
   COPY --from=base /app/.next/static ./.next/static

   EXPOSE 3000
   ENV PORT=3000
   CMD ["node", "server.js"]
   ```
2. Build and run:
   ```bash
   docker build -t gcp-frontend .
   docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://your-backend-ip:8000 gcp-frontend
   ```

---

## 2. Backend Deployment (FastAPI + PyTorch Model)

Since PyTorch utilizes significant RAM/CPU (and optionally GPUs), containerization is the standard pathway.

### Step 1: Dockerize the FastAPI Server
Create a `Dockerfile` inside the `backend/` directory.

> [!TIP]
> **Use CPU-Only PyTorch for Cloud Containers:**
> Standard PyTorch Docker images are very large (often 3GB+). If deploying to a CPU-only serverless platform (like AWS Fargate or Google Cloud Run), install the CPU-only version of PyTorch to reduce image size to ~800MB.

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (build-essential for numpy compiles if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and torchvision to keep image size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart gunicorn

# Copy code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run with Gunicorn using Uvicorn workers for production concurrency
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000"]
```

---

### Step 2: Choose Backend Cloud Hosting

#### Option A: Serverless Containers (Google Cloud Run / AWS Fargate)
Ideal for cost efficiency when traffic is variable.
1. Build and push the backend Docker image to Google Artifact Registry or AWS ECR.
2. Deploy the container.
3. Configure the container resource limits: **Minimum 2GB RAM** (3GB recommended) to comfortably hold PyTorch and EfficientNet weights in memory.

#### Option B: Virtual Machines with GPU (AWS EC2 g4dn.xlarge / Lambda Labs)
Required if low-latency processing (<200ms) or heavy batch requests are expected.
1. Rent a VM with an NVIDIA GPU (e.g. NVIDIA T4 or A10G).
2. Install Docker and NVIDIA Container Toolkit.
3. Replace the CPU PyTorch install command in the Dockerfile with standard CUDA PyTorch:
   `RUN pip install torch torchvision`
4. Run the docker container with `--gpus all` flags.

---

## 3. CORS and Domain configuration
Ensure your FastAPI `app.py` has the production frontend domain added to `allow_origins`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aeropoint.example.com", "https://your-vercel-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Use Nginx, Caddy, or an AWS Application Load Balancer (ALB) to handle SSL/HTTPS decryption before routing requests to port `8000` (FastAPI) and `3000` (Next.js).

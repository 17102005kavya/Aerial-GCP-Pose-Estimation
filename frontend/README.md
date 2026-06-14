# AeroPoint AI - GCP Pose Estimation Frontend

A professional, production-grade frontend dashboard built for Ground Control Point (GCP) Pose Estimation in high-resolution aerial imagery. This frontend connects to a FastAPI PyTorch backend running an EfficientNet-B3 multi-stage cascade regression model.

## Features

1. **Modern SaaS Landing Page & Hero Section:** Deep indigo/violet theme with subtle animation effects, detailed explanation of keypoint regression and cascade cropping, and fully responsive layouts.
2. **Interactive Workspace:**
   - **Premium Drag-and-Drop Uploader:** Supports JPG, JPEG, and PNG files with size limits (up to 50MB) and beautiful image previews.
   - **GCP Target Marker Overlay:** Draws an absolute circular target with animated crosshair pulses exactly over the predicted coordinate.
   - **Interactive Zoom & Pan Viewer:** Scroll wheel/pinch-to-zoom support (up to 15x) and mouse-dragging panning.
   - **Real-Time Coordinate Tracker:** Computes mouse hover coordinates relative to the image viewport and maps them back into the high-resolution original pixel space in real-time.
   - **Cascade Stages overlay:** Toggle to draw color-coded dashed boxes illustrating the exact crop windows (`[0, 1536, 768, 384]`) evaluated by each cascade stage of the EfficientNet model.
3. **Robust API Layer with Offline Fallback:** Custom API handler that queries the FastAPI backend, and automatically falls back to an in-browser mock simulation of the cascade sequence and regression values if the backend is unreachable.
4. **Theme Toggle:** Dark and Light mode theme switching powered by Tailwind CSS v4 and `next-themes`.

---

## Getting Started

### 1. Backend API Set Up (PyTorch Model Server)

Ensure you have Python 3.10+ installed. Navigate to the `backend/` directory:

```bash
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies (including torch, timm, fastapi, etc.)
pip install -r requirements.txt fastapi uvicorn python-multipart requests
```

Run the Uvicorn FastAPI server:
```bash
# Run server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
The server will initialize, download the base EfficientNet weights (if not cached), load the trained checkpoint `weights/best_pck.pth`, and listen at `http://localhost:8000`. You can inspect the interactive docs at `http://localhost:8000/docs`.

### 2. Frontend Set Up (Next.js Application)

Ensure you have Node.js 18+ and npm installed. Navigate to the `frontend/` directory:

```bash
# Install frontend dependencies
npm install

# Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Tech Stack

* **Framework:** Next.js 15 (App Router, Turbopack)
* **Language:** TypeScript (Strict Type Safety)
* **Styling:** Tailwind CSS v4
* **UI Components:** ShadCN UI & Base UI Radix primitives
* **Animations:** Framer Motion
* **API Handlers:** Fetch API client with automatic Simulation Fallback
* **Icons:** Lucide React
* **Confetti Animations:** Canvas Confetti

---

## File Structure

```
frontend/
├── public/                # Static assets
└── src/
    ├── app/
    │   ├── globals.css    # Global stylesheet and Tailwind v4 themes
    │   ├── layout.tsx     # Context wrapping providers (Theme, Query, Tooltip, Toaster)
    │   └── page.tsx       # Landing page and workspace wrapper
    ├── components/
    │   ├── ui/            # Reusable ShadCN primitive blocks (button, card, badge, tooltip, etc.)
    │   ├── Footer.tsx     # Footers
    │   ├── ImageViewer.tsx# Image Zoom, Pan, hover coordinates, and cascade overlays
    │   ├── Navbar.tsx     # Header bar with navigation and badges
    │   ├── PredictionCard.tsx # Statistics card for predicted coordinates and shapes
    │   ├── ThemeToggle.tsx# Swapper between Dark and Light mode
    │   └── UploadArea.tsx # Drag & Drop file uploader with previews
    ├── lib/
    │   ├── api.ts         # API Client and mock simulation generators
    │   └── utils.ts       # Tailwind class mergers
    └── providers/
        ├── QueryProvider.tsx # TanStack React Query provider
        └── ThemeProvider.tsx # next-themes stylesheet swapper
```

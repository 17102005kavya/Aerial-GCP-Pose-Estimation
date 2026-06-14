import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.predict import router as predict_router
from services.predictor import GCPPredictor

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("gcp_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load PyTorch model onto available device
    logger.info("Initializing backend server: loading PyTorch model...")
    try:
        app.state.predictor = GCPPredictor(
            config_path="configs/default.yaml",
            checkpoint_path="weights/best_pck.pth"
        )
        logger.info("PyTorch model loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Failed to load PyTorch model on startup: {e}")
        # We don't crash, but log it so requests will fail gracefully
        app.state.predictor = None
        
    yield
    
    # Shutdown: Clean up resources if necessary
    logger.info("Shutting down backend server...")
    if hasattr(app.state, "predictor"):
        del app.state.predictor

# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="GCP Pose Estimation API",
    description="Backend API running EfficientNet-B3 model for keypoint localization and shape classification of Ground Control Points (GCPs).",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
# Next.js will run on http://localhost:3000, so we allow it explicitly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount predict router
app.include_router(predict_router)

@app.get("/")
async def root():
    model_loaded = hasattr(app.state, "predictor") and app.state.predictor is not None
    return {
        "app": "GCP Pose Estimation Backend API",
        "status": "online",
        "model_loaded": model_loaded,
        "device": str(app.state.predictor.device) if model_loaded else "N/A"
    }

@app.get("/health")
async def health():
    model_loaded = hasattr(app.state, "predictor") and app.state.predictor is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

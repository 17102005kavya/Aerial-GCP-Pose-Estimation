import time
import io
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from PIL import Image
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter()

class CropWindow(BaseModel):
    left: int
    top: int
    width: int
    height: int

class Point(BaseModel):
    x: float
    y: float

class StageInfo(BaseModel):
    stage: int
    label: str
    crop_window: CropWindow
    predicted_normalized: Point
    predicted_absolute: Point

class PredictResponse(BaseModel):
    x: float
    y: float
    shape: str
    confidence: float
    filename: str
    width: int
    height: int
    inference_time_ms: float
    stages: List[StageInfo]

@router.post("/predict", response_model=PredictResponse)
async def predict_gcp(request: Request, file: UploadFile = File(...)):
    # 1. Validate file extension/type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only JPG, JPEG, and PNG files are supported."
        )

    try:
        # 2. Read file contents
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # 3. Convert grayscale or other modes to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        width, height = image.size
        
    except Exception as e:
        logger.error(f"Failed to open uploaded image: {e}")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # 4. Run prediction
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialized on the server.")

    start_time = time.time()
    try:
        prediction = predictor.predict(image)
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
        
    end_time = time.time()
    inference_time_ms = round((end_time - start_time) * 1000, 2)

    logger.info(
        f"Prediction complete for {file.filename} in {inference_time_ms}ms. "
        f"Result: shape={prediction['shape']}, coord=({prediction['x']}, {prediction['y']})"
    )

    return PredictResponse(
        x=prediction["x"],
        y=prediction["y"],
        shape=prediction["shape"],
        confidence=prediction["confidence"],
        filename=file.filename,
        width=width,
        height=height,
        inference_time_ms=inference_time_ms,
        stages=prediction["stages"]
    )

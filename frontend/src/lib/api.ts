export interface CropWindow {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface StageInfo {
  stage: number;
  label: string;
  crop_window: CropWindow;
  predicted_normalized: Point;
  predicted_absolute: Point;
}

export interface PredictResponse {
  x: number;
  y: number;
  shape: 'Cross' | 'L-Shaped' | 'Square';
  confidence: number;
  filename: string;
  width: number;
  height: number;
  inference_time_ms: number;
  stages: StageInfo[];
  isSimulation?: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Utility to read image dimensions from a File in the browser
 */
function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
      resolve({ width: img.width, height: img.height });
      URL.revokeObjectURL(img.src);
    };
    img.onerror = () => {
      resolve({ width: 4000, height: 3000 }); // Default fallback dimensions
    };
  });
}

/**
 * Generates a high-fidelity mock prediction response, simulating the cascade stages.
 */
async function simulatePrediction(file: File): Promise<PredictResponse> {
  const { width, height } = await getImageDimensions(file);
  
  // Select a realistic GCP center (slightly offset from the actual center)
  const targetX = Math.round(width * 0.35 + Math.random() * width * 0.3);
  const targetY = Math.round(height * 0.35 + Math.random() * height * 0.3);
  
  const shapes: Array<'Cross' | 'L-Shaped' | 'Square'> = ['Cross', 'L-Shaped', 'Square'];
  const shape = shapes[Math.floor(Math.random() * shapes.length)];
  const confidence = parseFloat((0.82 + Math.random() * 0.17).toFixed(4));
  
  const cascadeScales = [0, 1536, 768, 384];
  const stages: StageInfo[] = [];
  
  let currentCx = width / 2;
  let currentCy = height / 2;
  
  for (let i = 0; i < cascadeScales.length; i++) {
    const scale = cascadeScales[i];
    let left = 0;
    let top = 0;
    let cropW = width;
    let cropH = height;
    let label = 'Full Image';
    
    if (scale > 0) {
      label = `${scale}px Crop`;
      const cropSize = Math.min(scale, width, height);
      const half = Math.floor(cropSize / 2);
      
      left = Math.max(0, Math.min(Math.floor(currentCx - half), width - cropSize));
      top = Math.max(0, Math.min(Math.floor(currentCy - half), height - cropSize));
      cropW = cropSize;
      cropH = cropSize;
    }
    
    // Simulate model regression at this stage (approaching targetX/targetY)
    const noiseFactor = 0.15 / (i + 1); // Less noise at deeper stages
    const predictedAbsX = Math.round(targetX + (Math.random() - 0.5) * cropW * noiseFactor);
    const predictedAbsY = Math.round(targetY + (Math.random() - 0.5) * cropH * noiseFactor);
    
    // Normalized coordinates inside the crop window
    const xNorm = parseFloat(((predictedAbsX - left) / cropW).toFixed(4));
    const yNorm = parseFloat(((predictedAbsY - top) / cropH).toFixed(4));
    
    // Update center for next stage crop
    currentCx = predictedAbsX;
    currentCy = predictedAbsY;
    
    stages.push({
      stage: i,
      label,
      crop_window: { left, top, width: cropW, height: cropH },
      predicted_normalized: { x: xNorm, y: yNorm },
      predicted_absolute: { x: predictedAbsX, y: predictedAbsY }
    });
  }
  
  // The final coordinates are set to the last stage's predicted center
  const finalX = stages[stages.length - 1].predicted_absolute.x;
  const finalY = stages[stages.length - 1].predicted_absolute.y;
  
  // Artificial delay to simulate PyTorch processing latency
  await new Promise((resolve) => setTimeout(resolve, 800 + Math.random() * 600));
  
  return {
    x: finalX,
    y: finalY,
    shape,
    confidence,
    filename: file.name,
    width,
    height,
    inference_time_ms: parseFloat((100 + Math.random() * 150).toFixed(2)),
    stages,
    isSimulation: true,
  };
}

/**
 * Uploads an image file to the FastAPI /predict backend.
 * Automatically falls back to mock simulation if the backend is unreachable.
 */
export async function predictGcpImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    // Attempt real backend call
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 12000); // 12 second timeout
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`API Error (HTTP ${response.status}): ${errText || response.statusText}`);
    }
    
    return await response.json();
  } catch (error: any) {
    // Check if network error or timeout
    const isNetworkError = error.name === 'AbortError' || error.message?.includes('Failed to fetch') || error.message?.includes('NetworkError');
    
    if (isNetworkError) {
      console.warn('FastAPI backend is offline or unreachable. Falling back to local simulation mode...', error);
      return simulatePrediction(file);
    }
    
    // If it's a validation error or explicit 4xx/5xx from backend, rethrow it
    throw error;
  }
}

import sys
from pathlib import Path
import logging
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure the backend directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model, load_checkpoint
from src.dataset import IDX_TO_SHAPE, _to_tensor
from src.utils import load_config, get_device

logger = logging.getLogger(__name__)

class GCPPredictor:
    def __init__(self, config_path: str = "configs/default.yaml", checkpoint_path: str = "weights/best_pck.pth"):
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.cfg = load_config(config_path)
        self.img_size = self.cfg.get("model", {}).get("img_size", 512)
        self.cascade_scales = self.cfg.get("inference", {}).get("cascade_scales", [0, 1536, 768, 384])
        self.use_tta = self.cfg.get("inference", {}).get("tta", True)
        
        logger.info(f"Model config: img_size={self.img_size}, cascade_scales={self.cascade_scales}, TTA={self.use_tta}")
        
        # Build and load model
        self.model = build_model(self.cfg.get("model", {}), self.device)
        self.model = load_checkpoint(self.model, checkpoint_path, self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def _flip_coords(self, coords: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
        out = coords.copy()
        if flip_h:
            out[:, 0] = 1.0 - out[:, 0]
        if flip_v:
            out[:, 1] = 1.0 - out[:, 1]
        return out

    def _tta_predict(self, img_tensor: torch.Tensor) -> tuple:
        """Returns (coords [1,2] normalized, shape_probs [1,3] softmax-averaged)."""
        all_coords = np.zeros((1, 2), dtype=np.float32)
        all_probs  = np.zeros((1, 3), dtype=np.float32)

        for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
            batch = img_tensor.clone()
            if flip_h:
                batch = torch.flip(batch, dims=[3])
            if flip_v:
                batch = torch.flip(batch, dims=[2])
            batch = batch.to(self.device)
            
            with torch.no_grad():
                coords, logits = self.model(batch)
            
            coords_np = self._flip_coords(coords.cpu().numpy(), flip_h, flip_v)
            all_coords += coords_np
            all_probs  += F.softmax(logits, dim=-1).cpu().numpy()

        all_coords /= 4
        all_probs  /= 4
        return all_coords, all_probs

    def _predict_single(self, img_tensor: torch.Tensor) -> tuple:
        """Returns (coords [1,2] normalized, shape_probs [1,3] softmax)."""
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            coords, logits = self.model(img_tensor)
        return coords.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        """
        Runs multi-stage cascade inference on a PIL Image.
        Returns:
            {
                "x": float (pixel coordinate on original image),
                "y": float (pixel coordinate on original image),
                "shape": str (predicted shape name: "Cross", "L-Shaped", "Square"),
                "confidence": float (shape probability),
                "stages": list (details about each cascade stage for visualization/debug)
            }
        """
        W, H = img.size
        # Start at the center of the image
        cx, cy = W / 2.0, H / 2.0
        
        centers = None
        stages_info = []
        final_probs = None
        
        for stage_idx, scale in enumerate(self.cascade_scales):
            if scale <= 0 or centers is None:
                # Stage 0: full image
                crop = img
                left, top = 0, 0
                crop_w, crop_h = W, H
                crop_label = "Full Image"
            else:
                cx_val, cy_val = centers["x"], centers["y"]
                crop_size = min(scale, W, H)
                half = crop_size // 2
                
                left = int(np.clip(cx_val - half, 0, max(0, W - crop_size)))
                top  = int(np.clip(cy_val - half, 0, max(0, H - crop_size)))
                right  = min(W, left + crop_size)
                bottom = min(H, top + crop_size)
                
                crop = img.crop((left, top, right, bottom))
                crop_w, crop_h = crop.size
                crop_label = f"{scale}px Crop"
                
            resized = crop.resize((self.img_size, self.img_size), Image.BILINEAR)
            tensor = _to_tensor(resized).unsqueeze(0)  # [1, 3, 512, 512]
            
            # Predict coords and shape logits
            if self.use_tta:
                coords, probs = self._tta_predict(tensor)
            else:
                coords, probs = self._predict_single(tensor)
                
            x_norm, y_norm = coords[0]
            x_px = left + x_norm * crop_w
            y_px = top  + y_norm * crop_h
            
            centers = {"x": float(x_px), "y": float(y_px)}
            final_probs = probs[0]
            
            stages_info.append({
                "stage": stage_idx,
                "label": crop_label,
                "crop_window": {"left": left, "top": top, "width": crop_w, "height": crop_h},
                "predicted_normalized": {"x": float(x_norm), "y": float(y_norm)},
                "predicted_absolute": {"x": float(x_px), "y": float(y_px)}
            })
            
        shape_idx = int(np.argmax(final_probs))
        shape_name = IDX_TO_SHAPE[shape_idx]
        confidence = float(final_probs[shape_idx])
        
        return {
            "x": round(centers["x"], 2),
            "y": round(centers["y"], 2),
            "shape": shape_name,
            "confidence": round(confidence, 4),
            "stages": stages_info
        }

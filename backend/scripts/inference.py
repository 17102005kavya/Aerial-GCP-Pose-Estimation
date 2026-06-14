#!/usr/bin/env python3
"""
inference.py – Coarse-to-fine cascade inference.

The model is trained with multi-scale crops (full image down to tight
crops, see configs/default.yaml `data.scale_choices`), so it has learned
to localize the GCP marker at ANY zoom level. This lets a single checkpoint
drive a cascade:

    Stage 0 (coarse) : Resize the FULL image to img_size, run the model ->
                        rough (x, y) estimate in original pixel space.
    Stage 1..N (refine): Crop a window of `cascade_scales[i]` px around the
                        previous stage's estimate (from the ORIGINAL image),
                        resize to img_size, run the model again -> tighter
                        estimate. Each stage zooms in further.

The final stage's shape-classification logits are used for the predicted
`verified_shape` (highest-resolution view = most reliable classification).

Default cascade (`configs/default.yaml` -> inference.cascade_scales):
    [0, 1536, 768, 384]   # full image -> 1536px -> 768px -> 384px

Usage:
    python scripts/inference.py \
        --data_root /path/to/test_dataset \
        --checkpoint ./runs/exp1/best_pck.pth \
        --output predictions.json \
        --config configs/default.yaml \
        --tta
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.dataset import IDX_TO_SHAPE, _to_tensor
from src.model import build_model, load_checkpoint
from src.utils import get_device, load_config, save_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Generic crop-and-resize dataset for one cascade stage
# ---------------------------------------------------------------------------

class CascadeStageDataset(Dataset):
    """
    For each image, crops a `scale` x `scale` window centered on the
    previous stage's (x, y) estimate (in original pixel space), then
    resizes to img_size to match the training distribution.

    `scale == 0` means "use the full image" (Stage 0 / coarse pass).
    `centers` is None for Stage 0.
    """

    def __init__(
        self,
        root: str,
        samples: List[str],
        img_size: int,
        scale: int,
        centers: Optional[Dict[str, dict]] = None,
    ):
        self.root = Path(root)
        self.samples = samples
        self.img_size = img_size
        self.scale = scale
        self.centers = centers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        W, H = img.size

        if self.scale <= 0 or self.centers is None:
            # Stage 0: full image
            crop = img
            left, top = 0, 0
            crop_w, crop_h = W, H
        else:
            c = self.centers[rel_path]
            cx, cy = c["x"], c["y"]
            crop_size = min(self.scale, W, H)
            half = crop_size // 2
            left = int(np.clip(cx - half, 0, max(0, W - crop_size)))
            top  = int(np.clip(cy - half, 0, max(0, H - crop_size)))
            right  = min(W, left + crop_size)
            bottom = min(H, top + crop_size)
            crop = img.crop((left, top, right, bottom))
            crop_w, crop_h = crop.size

        resized = crop.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = _to_tensor(resized)

        return tensor, rel_path, left, top, crop_w, crop_h, W, H


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def _flip_coords(coords: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    out = coords.copy()
    if flip_h:
        out[:, 0] = 1.0 - out[:, 0]
    if flip_v:
        out[:, 1] = 1.0 - out[:, 1]
    return out


def tta_predict(model, imgs: torch.Tensor, device: torch.device):
    """Returns (coords [B,2] normalized, shape_probs [B,3] softmax-averaged)."""
    B = imgs.shape[0]
    all_coords = np.zeros((B, 2), dtype=np.float32)
    all_probs  = np.zeros((B, 3), dtype=np.float32)

    for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
        batch = imgs.clone()
        if flip_h:
            batch = torch.flip(batch, dims=[3])
        if flip_v:
            batch = torch.flip(batch, dims=[2])
        batch = batch.to(device)
        with torch.no_grad():
            coords, logits = model(batch)
        coords_np = _flip_coords(coords.cpu().numpy(), flip_h, flip_v)
        all_coords += coords_np
        all_probs  += F.softmax(logits, dim=-1).cpu().numpy()

    all_coords /= 4
    all_probs  /= 4
    return all_coords, all_probs


def predict(model, imgs: torch.Tensor, device: torch.device):
    """Returns (coords [B,2] normalized, shape_probs [B,3] softmax)."""
    imgs = imgs.to(device)
    with torch.no_grad():
        coords, logits = model(imgs)
    return coords.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# One cascade stage
# ---------------------------------------------------------------------------

def run_stage(
    model,
    root: str,
    samples: List[str],
    img_size: int,
    scale: int,
    centers: Optional[Dict[str, dict]],
    device,
    batch_size: int,
    use_tta: bool,
    stage_idx: int,
) -> Dict[str, dict]:
    """
    Runs one cascade stage. Returns dict:
        rel_path -> {"x": px, "y": px, "shape_probs": [p_cross, p_lshaped, p_square]}
    x/y are in ORIGINAL image pixel space.
    """
    dataset = CascadeStageDataset(root, samples, img_size, scale, centers)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    label = "full image" if scale <= 0 or centers is None else f"{scale}px crop"
    logger.info(f"[Stage {stage_idx}] {label} — {len(dataset)} images...")

    results = {}
    for imgs, paths, lefts, tops, crop_ws, crop_hs, Ws, Hs in loader:
        if use_tta:
            coords, probs = tta_predict(model, imgs, device)
        else:
            coords, probs = predict(model, imgs, device)

        for i in range(imgs.shape[0]):
            x_norm, y_norm = coords[i]
            left, top = int(lefts[i]), int(tops[i])
            crop_w, crop_h = int(crop_ws[i]), int(crop_hs[i])

            x_px = left + x_norm * crop_w
            y_px = top  + y_norm * crop_h

            results[paths[i]] = {
                "x": float(x_px),
                "y": float(y_px),
                "W": int(Ws[i]),
                "H": int(Hs[i]),
                "shape_probs": probs[i],
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output",      default="predictions.json")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--cascade_scales", type=int, nargs="+", default=None,
                   help="Crop sizes (px, original image) per cascade stage; "
                        "first must be 0 (full image). Overrides config.")
    p.add_argument("--tta", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config)
    img_size = cfg.get("model", {}).get("img_size", 512)
    device = get_device()

    cascade_scales = args.cascade_scales or cfg.get("inference", {}).get(
        "cascade_scales", [0, 1536, 768, 384]
    )
    if cascade_scales[0] != 0:
        raise ValueError("cascade_scales[0] must be 0 (full-image coarse pass)")

    use_tta = args.tta or cfg.get("inference", {}).get("tta", False)

    logger.info(
        f"Device: {device} | img_size: {img_size} | "
        f"cascade_scales: {cascade_scales} | TTA: {use_tta}"
    )

    model = build_model(cfg.get("model", {}), device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Discover test samples
    root = Path(args.data_root)
    samples = sorted(str(p.relative_to(root)) for p in root.rglob("*.JPG"))
    logger.info(f"Found {len(samples)} test images")

    # ---------------------------------------------------------------
    # Run cascade
    # ---------------------------------------------------------------
    centers = None
    stage_results = None
    for stage_idx, scale in enumerate(cascade_scales):
        stage_results = run_stage(
            model, args.data_root, samples, img_size, scale, centers,
            device, args.batch_size, use_tta, stage_idx,
        )
        centers = stage_results  # next stage crops around this stage's (x, y)

    # ---------------------------------------------------------------
    # Final predictions (last stage's coords + shape classification)
    # ---------------------------------------------------------------
    predictions = {}
    for rel_path, res in stage_results.items():
        shape_idx = int(np.argmax(res["shape_probs"]))
        predictions[rel_path] = {
            "mark": {"x": round(res["x"], 2), "y": round(res["y"], 2)},
            "verified_shape": IDX_TO_SHAPE[shape_idx],
        }

    save_json(predictions, args.output)
    logger.info(f"Done. {len(predictions)} predictions written to {args.output}")


if __name__ == "__main__":
    main()

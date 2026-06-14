"""
dataset.py – GCP dataset with multi-scale, coordinate-aware augmentations.

Key fix vs. v1:
    Training used ONLY tight crops (crop_size=384, crop_margin=128), so the
    model only ever saw the GCP marker filling most of the frame. At
    inference, a coarse full-image pass produced near-meaningless coordinates
    because the model had never seen that distribution.

    Fix — multi-scale training: for every sample, randomly choose a crop
    scale from `scale_choices` (0 = full image, otherwise a square crop of
    that pixel size around the GCP, with random jitter so the marker is not
    always dead-center). The SAME model therefore learns both coarse
    full-image localization and fine-grained refinement, and is directly
    usable for coarse-to-fine / sliding-window inference.

Each sample returns:
    image  : FloatTensor [3, H, W]  (normalized)
    coords : FloatTensor [2]        (x, y) in [0, 1] relative to the crop
    label  : LongTensor  []         shape class index (0=Cross, 1=L-Shaped, 2=Square)
    meta   : dict                   original path + scale factors for un-normalizing
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler

SHAPE_TO_IDX = {"Cross": 0, "L-Shaped": 1, "Square": 2}
IDX_TO_SHAPE = {v: k for k, v in SHAPE_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Label normalization + manual patches
# ---------------------------------------------------------------------------
# The raw curated_gcp_marks.json contains a few inconsistent shape strings
# and is missing the verified_shape for a handful of entries entirely.
# Both issues are baked in here so every consumer of `load_labels` gets
# clean, consistent labels without re-running notebook cells.

# Map raw / inconsistent shape strings -> canonical SHAPE_TO_IDX keys.
SHAPE_NORMALIZATION = {
    "L-Shape": "L-Shaped",
    "l-shaped": "L-Shaped",
    "l-shape": "L-Shaped",
    "L shaped": "L-Shaped",
    "L Shape": "L-Shaped",
    "square": "Square",
    "Squares": "Square",
    "cross": "Cross",
    "Crosses": "Cross",
}

# Entries with no `verified_shape` in the source JSON at all. Manually
# reviewed and labeled (see README "Data Quality Issues").
MANUAL_SHAPE_LABELS = {
    "Seashell Ras el Hekma/Survey 3/GCP12/DJI_20240605112759_0254.JPG": "Square",
    "Vedanta GOA Bicholim/MCDR 2024/GCP32/19_4_DJI_0066.JPG": "L-Shaped",
    "Vedanta GOA Bicholim/MCDR 2024/GCP17/12_2_DJI_0558.JPG": "L-Shaped",
    "UTCL UNCL Additional Area/Survey-1/GCP-98/DJI_20240425131303_0192_V.JPG": "L-Shaped",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def _rotate_coord(x: float, y: float, cx: float, cy: float, deg: float):
    rad = math.radians(deg)
    dx, dy = x - cx, y - cy
    nx = dx * math.cos(rad) - dy * math.sin(rad) + cx
    ny = dx * math.sin(rad) + dy * math.cos(rad) + cy
    return nx, ny


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

class GCPAugmenter:
    """
    Applies a sequence of augmentations that also transform the keypoint coords.
    All operations work on PIL images and (x, y) pixel coordinates.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __call__(
        self,
        img: Image.Image,
        x: float,
        y: float,
    ) -> Tuple[Image.Image, float, float]:
        W, H = img.size

        # Random horizontal flip
        if random.random() < self.cfg.get("hflip_prob", 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x = W - x

        # Random vertical flip
        if random.random() < self.cfg.get("vflip_prob", 0.5):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            y = H - y

        # Random rotation
        deg = random.uniform(
            -self.cfg.get("rotation_degrees", 30),
             self.cfg.get("rotation_degrees", 30),
        )
        img = img.rotate(deg, resample=Image.BILINEAR, expand=False)
        cx, cy = W / 2.0, H / 2.0
        x, y = _rotate_coord(x, y, cx, cy, -deg)  # PIL rotates CCW

        # Random scale crop
        lo, hi = self.cfg.get("scale_range", [0.7, 1.0])
        scale = random.uniform(lo, hi)
        nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
        # Clamp crop so GCP stays inside
        margin = 8
        max_left = max(0, int(x) - margin)
        max_top  = max(0, int(y) - margin)
        left = random.randint(0, max(0, min(max_left, W - nw)))
        top  = random.randint(0, max(0, min(max_top, H - nh)))
        img = img.crop((left, top, left + nw, top + nh))
        img = img.resize((W, H), Image.BILINEAR)
        x = (x - left) * (W / nw)
        y = (y - top)  * (H / nh)

        # Color jitter
        jcfg = self.cfg.get("color_jitter", {})
        if jcfg:
            from PIL import ImageEnhance
            for fn, key in [
                (ImageEnhance.Brightness, "brightness"),
                (ImageEnhance.Contrast,   "contrast"),
                (ImageEnhance.Color,      "saturation"),
            ]:
                v = jcfg.get(key, 0.0)
                if v > 0:
                    factor = random.uniform(1 - v, 1 + v)
                    img = fn(img).enhance(factor)
            hue = jcfg.get("hue", 0.0)
            if hue > 0 and random.random() < 0.3:
                # Slight hue shift via HSV manipulation
                arr = np.array(img.convert("HSV"), dtype=np.float32)
                arr[..., 0] = (arr[..., 0] + random.uniform(-hue * 180, hue * 180)) % 256
                img = Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB")

        # Gaussian blur
        if random.random() < self.cfg.get("blur_prob", 0.2):
            k = self.cfg.get("blur_kernel", 5)
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, k / 2)))

        # Random grayscale
        if random.random() < self.cfg.get("grayscale_prob", 0.05):
            img = img.convert("L").convert("RGB")

        # Clamp coords
        W2, H2 = img.size
        x = max(0.0, min(float(W2 - 1), x))
        y = max(0.0, min(float(H2 - 1), y))

        return img, x, y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GCPDataset(Dataset):
    """
    Loads images and labels for GCP keypoint + shape classification.

    Multi-scale crop sampling (training):
        For each sample, a crop scale is randomly chosen from
        `scale_choices` (pixel size of the square crop, 0 = full image).
        A crop of that size is taken around the GCP center with up to
        `crop_jitter` px of random offset (so the marker is not always
        dead-center — this matches what a sliding-window / coarse Stage-1
        pass will actually feed the model at inference time).

    Fixed-scale mode (validation / single-scale eval):
        Pass `scale_choices=[val_scale]` (and crop_jitter=0 for a
        deterministic, comparable validation signal across epochs).

    Args:
        root: Path to dataset root (train or test).
        labels: Dict mapping relative path -> {"mark": {x, y}, "verified_shape": str}.
                Pass None for test set (no labels available).
        img_size: Final image size after all transforms (model input).
        scale_choices: List of crop sizes (px) to sample from. 0 = full image.
        crop_jitter: Max random pixel offset applied to the crop center.
        augment: Whether to apply training augmentations.
        aug_cfg: Augmentation config dict.
    """

    def __init__(
        self,
        root: str,
        labels: Optional[Dict],
        img_size: int = 512,
        scale_choices: Optional[List[int]] = None,
        crop_jitter: int = 0,
        augment: bool = False,
        aug_cfg: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.labels = labels
        self.img_size = img_size
        self.scale_choices = scale_choices if scale_choices is not None else [0]
        self.crop_jitter = crop_jitter
        self.augment = augment
        self.augmenter = GCPAugmenter(aug_cfg or {}) if augment else None

        # Build sample list
        if labels is not None:
            self.samples = list(labels.keys())
        else:
            # Test set: discover all JPGs recursively
            self.samples = [
                str(p.relative_to(self.root))
                for p in sorted(self.root.rglob("*.JPG"))
            ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path = self.samples[idx]
        img_path = self.root / rel_path
        img = Image.open(img_path).convert("RGB")
        W_orig, H_orig = img.size

        if self.labels is not None:
            ann = self.labels[rel_path]
            x_abs = float(ann["mark"]["x"])
            y_abs = float(ann["mark"]["y"])
            shape_idx = SHAPE_TO_IDX[ann["verified_shape"]]
        else:
            x_abs, y_abs = W_orig / 2.0, H_orig / 2.0  # placeholder for test
            shape_idx = -1

        # ------------------------------------------------------------
        # Multi-scale crop selection
        # ------------------------------------------------------------
        scale = random.choice(self.scale_choices)

        if scale > 0 and self.labels is not None:
            crop_size = min(scale, W_orig, H_orig)
            half = crop_size // 2

            # Jitter the crop center so the marker isn't always centered.
            jx = random.randint(-self.crop_jitter, self.crop_jitter) if self.crop_jitter > 0 else 0
            jy = random.randint(-self.crop_jitter, self.crop_jitter) if self.crop_jitter > 0 else 0
            cx = x_abs + jx
            cy = y_abs + jy

            left = int(np.clip(cx - half, 0, max(0, W_orig - crop_size)))
            top  = int(np.clip(cy - half, 0, max(0, H_orig - crop_size)))
            right  = left + crop_size
            bottom = top + crop_size

            img = img.crop((left, top, right, bottom))
            x_abs -= left
            y_abs -= top
        # scale == 0 (or test set): use the full image as-is

        # Resize to model input size
        W_crop, H_crop = img.size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        scale_x = self.img_size / W_crop
        scale_y = self.img_size / H_crop
        x_abs *= scale_x
        y_abs *= scale_y

        # Augmentations (training only)
        if self.augment and self.augmenter is not None:
            img, x_abs, y_abs = self.augmenter(img, x_abs, y_abs)

        # Normalize coords to [0, 1]
        x_norm = x_abs / self.img_size
        y_norm = y_abs / self.img_size
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))

        tensor = _to_tensor(img)
        coords = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        label  = torch.tensor(shape_idx, dtype=torch.long)

        meta = {
            "path": rel_path,
            "orig_w": W_orig,
            "orig_h": H_orig,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "crop_scale": scale,
        }

        return tensor, coords, label, meta


# ---------------------------------------------------------------------------
# Weighted sampler for class imbalance
# ---------------------------------------------------------------------------

def build_weighted_sampler(dataset: GCPDataset) -> WeightedRandomSampler:
    """Creates a sampler that up-weights under-represented shape classes."""
    labels = [
        SHAPE_TO_IDX[dataset.labels[path]["verified_shape"]]
        for path in dataset.samples
    ]
    class_counts = np.bincount(labels, minlength=3).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def load_labels(json_path: str) -> Dict:
    """
    Load curated_gcp_marks.json and apply label cleaning:

      1. Normalize inconsistent `verified_shape` strings (e.g. "L-Shape" ->
         "L-Shaped") via SHAPE_NORMALIZATION.
      2. Fill in the 4 entries missing `verified_shape` entirely using the
         manually-reviewed MANUAL_SHAPE_LABELS.

    Any entry that still doesn't have a valid verified_shape afterwards is
    dropped (with a printed warning) rather than crashing downstream.
    """
    with open(json_path) as f:
        labels = json.load(f)

    cleaned = {}
    for path, ann in labels.items():
        shape = ann.get("verified_shape")

        if shape is None or shape not in SHAPE_TO_IDX:
            if shape in SHAPE_NORMALIZATION:
                shape = SHAPE_NORMALIZATION[shape]
            elif path in MANUAL_SHAPE_LABELS:
                shape = MANUAL_SHAPE_LABELS[path]
            else:
                print(f"[load_labels] WARNING: dropping '{path}' — "
                      f"no valid verified_shape ('{ann.get('verified_shape')}')")
                continue

        ann = dict(ann)
        ann["verified_shape"] = shape
        cleaned[path] = ann

    return cleaned


def train_val_split(
    labels: Dict,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    paths = list(labels.keys())
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio))
    val_paths  = set(paths[:n_val])
    train_paths = set(paths[n_val:])
    return (
        {k: v for k, v in labels.items() if k in train_paths},
        {k: v for k, v in labels.items() if k in val_paths},
    )

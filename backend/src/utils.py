"""
utils.py – Shared utilities: config, logging, JSON I/O, reproducibility.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    if not _HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge overrides into base config."""
    result = base.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str, level: int = logging.INFO):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "train.log"),
        ],
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CuDNN (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Dict, path: str, indent: int = 2):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
    logging.getLogger(__name__).info(f"Saved JSON → {path}")


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Denormalize coordinates
# ---------------------------------------------------------------------------

def denorm_coords(
    x_norm: float,
    y_norm: float,
    orig_w: int,
    orig_h: int,
    scale_x: float,
    scale_y: float,
    img_size: int,
) -> tuple:
    """
    Convert normalized model output coords back to original image pixel coords.

    The pipeline is:
        orig_image -> crop (left, top) -> resize (img_size) -> normalize [0,1]
    This function inverts the resize+normalize step.
    The crop offset must be added separately by the caller.
    """
    x_px = x_norm * img_size / scale_x
    y_px = y_norm * img_size / scale_y
    return x_px, y_px

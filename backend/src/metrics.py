"""
metrics.py – Evaluation metrics for GCP pose estimation.

Metrics:
    PCK@K  : % of keypoints within K pixels of ground truth
    Macro F1 : unweighted mean F1 across all shape classes
"""

from typing import Dict, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# PCK (Percentage of Correct Keypoints)
# ---------------------------------------------------------------------------

def compute_pck(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    img_size: int,
    thresholds: List[int] = (10, 25, 50),
) -> Dict[str, float]:
    """
    Computes PCK at multiple pixel thresholds.

    Args:
        pred_coords : [N, 2] predicted (x, y) in normalized [0,1] space
        gt_coords   : [N, 2] ground truth in [0,1]
        img_size    : image size (to convert normalized → pixels)
        thresholds  : list of pixel thresholds

    Returns:
        dict: {'pck10': ..., 'pck25': ..., 'pck50': ...}
    """
    # Convert to pixel space
    pred_px = pred_coords * img_size
    gt_px   = gt_coords   * img_size

    # Euclidean distance per sample
    dist = np.sqrt(((pred_px - gt_px) ** 2).sum(axis=-1))   # [N]

    results = {}
    for t in thresholds:
        results[f"pck{t}"] = float((dist <= t).mean())

    results["mean_dist_px"] = float(dist.mean())
    return results


# ---------------------------------------------------------------------------
# Macro F1
# ---------------------------------------------------------------------------

def compute_macro_f1(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Computes per-class and macro F1 score.

    Args:
        pred_labels : [N] predicted class indices
        gt_labels   : [N] ground truth class indices
        num_classes : number of classes

    Returns:
        dict with per-class F1 + 'macro_f1'
    """
    f1s = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (gt_labels == c)).sum()
        fp = ((pred_labels == c) & (gt_labels != c)).sum()
        fn = ((pred_labels != c) & (gt_labels == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(float(f1))

    from src.dataset import IDX_TO_SHAPE
    per_class = {f"f1_{IDX_TO_SHAPE[i]}": f1s[i] for i in range(num_classes)}
    per_class["macro_f1"] = float(np.mean(f1s))
    return per_class


# ---------------------------------------------------------------------------
# Accumulator for streaming metric computation
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """Collects batch-level predictions and computes epoch-level metrics."""

    def __init__(self, img_size: int):
        self.img_size = img_size
        self._pred_coords: List[np.ndarray] = []
        self._gt_coords:   List[np.ndarray] = []
        self._pred_labels: List[np.ndarray] = []
        self._gt_labels:   List[np.ndarray] = []

    def update(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_labels: torch.Tensor,
    ):
        self._pred_coords.append(pred_coords.detach().cpu().numpy())
        self._gt_coords.append(gt_coords.detach().cpu().numpy())
        self._pred_labels.append(pred_logits.argmax(dim=-1).detach().cpu().numpy())
        self._gt_labels.append(gt_labels.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        pred_coords = np.concatenate(self._pred_coords, axis=0)
        gt_coords   = np.concatenate(self._gt_coords,   axis=0)
        pred_labels = np.concatenate(self._pred_labels, axis=0)
        gt_labels   = np.concatenate(self._gt_labels,   axis=0)

        pck  = compute_pck(pred_coords, gt_coords, self.img_size)
        f1   = compute_macro_f1(pred_labels, gt_labels)
        return {**pck, **f1}

    def reset(self):
        self._pred_coords.clear()
        self._gt_coords.clear()
        self._pred_labels.clear()
        self._gt_labels.clear()

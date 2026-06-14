"""
losses.py – Loss functions for GCP multi-task learning.

Wing Loss (Feng et al., 2018) for keypoint regression:
    Better than MSE/L1 for fine-grained localization — applies a log
    penalty for small residuals and linear for large ones, which drives
    the model to be precise at the tight PCK@10px threshold.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Wing Loss
# ---------------------------------------------------------------------------

class WingLoss(nn.Module):
    """
    Wing loss for 2D keypoint regression.

    L(x) = w * ln(1 + |x| / eps)       if |x| < w
           |x| - C                       otherwise
    where C = w - w * ln(1 + w / eps)  (continuity constant)

    Args:
        w:   Width of the non-linear region (default 10.0 px).
        eps: Curvature of the log region   (default 2.0).
    """

    def __init__(self, w: float = 10.0, eps: float = 2.0):
        super().__init__()
        self.w   = w
        self.eps = eps
        self.C   = w - w * math.log(1.0 + w / eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, 2] predicted (x, y) in normalized [0,1] space
            target : [B, 2] ground truth coords in [0,1]
        Returns:
            scalar loss
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.eps),
            diff - self.C,
        )
        return loss.mean()


# ---------------------------------------------------------------------------
# Focal Cross-Entropy for class imbalance
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Down-weights well-classified examples so the model focuses on hard ones.
    Useful for imbalanced shape classes (L-Shaped is rare).

    Args:
        gamma : Focusing parameter (default 2.0).
        label_smoothing: Label smoothing applied before focal weighting.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        # Label-smoothed NLL
        n_cls = logits.size(1)
        smooth_targets = torch.full_like(log_probs, self.label_smoothing / (n_cls - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        nll = -(smooth_targets * log_probs).sum(dim=-1)
        # Focal weight
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t) ** self.gamma
        return (focal * nll).mean()


# ---------------------------------------------------------------------------
# Joint multi-task loss
# ---------------------------------------------------------------------------

class JointLoss(nn.Module):
    """
    Combined loss for keypoint regression + shape classification.

    L = WingLoss(keypoints) + λ * FocalLoss(shape)

    The lambda weight is tunable; default 0.5 works well empirically.
    """

    def __init__(
        self,
        wing_w: float = 10.0,
        wing_eps: float = 2.0,
        lambda_cls: float = 0.5,
        label_smoothing: float = 0.1,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.wing     = WingLoss(wing_w, wing_eps)
        self.focal    = FocalLoss(gamma, label_smoothing)
        self.lambda_cls = lambda_cls

    def forward(
        self,
        pred_coords: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            pred_coords : [B, 2] predicted normalized coordinates
            pred_logits : [B, 3] shape class logits
            gt_coords   : [B, 2] ground truth normalized coordinates
            gt_labels   : [B]   ground truth shape class indices
        Returns:
            dict with keys: total, kp_loss, cls_loss
        """
        kp_loss  = self.wing(pred_coords, gt_coords)
        cls_loss = self.focal(pred_logits, gt_labels)
        total    = kp_loss + self.lambda_cls * cls_loss
        return {"total": total, "kp_loss": kp_loss, "cls_loss": cls_loss}

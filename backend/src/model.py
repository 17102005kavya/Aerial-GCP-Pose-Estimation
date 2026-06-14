"""
model.py – Multi-task GCP model.

Architecture:
    Backbone : EfficientNet-B3 (pretrained ImageNet)
    Neck     : Global Average Pooling + Dropout
    Head 1   : Keypoint regression  -> (x, y) normalized [0,1]
    Head 2   : Shape classification -> 3-class logits
"""

from typing import Tuple

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# ---------------------------------------------------------------------------
# Attention pooling (optional, replaces GAP for richer spatial features)
# ---------------------------------------------------------------------------

class SpatialAttentionPool(nn.Module):
    """
    Learns a soft attention map over the spatial feature grid,
    then returns the weighted-sum feature vector.
    Useful when the GCP may not be perfectly centred in the crop.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        weights = self.attn(x).view(B, 1, H * W)          # [B, 1, HW]
        feats   = x.view(B, C, H * W)                      # [B, C, HW]
        pooled  = (feats * weights).sum(dim=-1)             # [B, C]
        return pooled


# ---------------------------------------------------------------------------
# Regression head (keypoint)
# ---------------------------------------------------------------------------

class KeypointHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),   # output in (0, 1) — normalized coords
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, 2]


# ---------------------------------------------------------------------------
# Classification head (shape)
# ---------------------------------------------------------------------------

class ShapeHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, 3]


# ---------------------------------------------------------------------------
# Full multi-task model
# ---------------------------------------------------------------------------

class GCPModel(nn.Module):
    """
    Multi-task model for GCP keypoint localization + shape classification.

    Args:
        backbone_name: timm model name (default 'efficientnet_b3').
        pretrained: Load ImageNet weights.
        num_classes: Number of shape classes (default 3).
        dropout: Dropout rate in MLP heads.
        use_attn_pool: Use spatial attention pooling instead of GAP.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_attn_pool: bool = False,
    ):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required: pip install timm")

        # Load backbone without classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove classifier
            global_pool="",         # remove pooling (we do it ourselves)
        )
        # Infer feature dimension by a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)   # [1, C, H, W]
            feat_channels = feats.shape[1]

        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            self.pool = SpatialAttentionPool(feat_channels)
        else:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.dropout = nn.Dropout(dropout)
        self.keypoint_head = KeypointHead(feat_channels, dropout)
        self.shape_head    = ShapeHead(feat_channels, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)          # [B, C, H, W]
        pooled = self.pool(feats)         # [B, C]
        pooled = self.dropout(pooled)

        coords = self.keypoint_head(pooled)   # [B, 2] in (0,1)
        logits = self.shape_head(pooled)      # [B, 3]
        return coords, logits

    def freeze_backbone(self):
        """Freeze backbone weights (useful for initial warmup)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone weights for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> GCPModel:
    model = GCPModel(
        backbone_name=cfg.get("backbone", "efficientnet_b3"),
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.3),
        use_attn_pool=cfg.get("use_attn_pool", False),
    )
    return model.to(device)


def load_checkpoint(model: GCPModel, path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    return model

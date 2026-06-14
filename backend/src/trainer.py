"""
trainer.py – Training loop with mixed precision, LR scheduling, and checkpointing.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.losses import JointLoss
from src.metrics import MetricAccumulator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup_epochs:
            factor = (e + 1) / self.warmup_epochs
            for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base * factor
        else:
            self.cosine.step()
        self.current_epoch += 1

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        output_dir: str,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg.get("training", {})
        lcfg = cfg.get("loss", {})

        self.epochs = tcfg.get("epochs", 60)
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.mixed_precision = tcfg.get("mixed_precision", True) and device.type == "cuda"
        self.img_size = cfg.get("model", {}).get("img_size", 512)
        self.early_stopping_patience = tcfg.get("early_stopping_patience", 10)
        self.monitor_metric = tcfg.get("early_stopping_metric", "pck25")

        # Loss
        self.criterion = JointLoss(
            wing_w=lcfg.get("wing_w", 10.0),
            wing_eps=lcfg.get("wing_eps", 2.0),
            lambda_cls=lcfg.get("lambda_cls", 0.5),
            label_smoothing=lcfg.get("label_smoothing", 0.1),
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=tcfg.get("lr", 1e-4),
            weight_decay=tcfg.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=tcfg.get("warmup_epochs", 5),
            total_epochs=self.epochs,
        )

        self.scaler = GradScaler() if self.mixed_precision else None

        # Tracking
        self.best_metric = -1.0
        self.best_metric_name = self.monitor_metric
        self.patience_counter = 0
        self.history: Dict[str, list] = {}

    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        accumulator = MetricAccumulator(self.img_size)
        total_loss = kp_loss_sum = cls_loss_sum = 0.0
        n_batches = 0

        for imgs, coords, labels, _ in loader:
            imgs   = imgs.to(self.device)
            coords = coords.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.mixed_precision):
                pred_coords, pred_logits = self.model(imgs)
                losses = self.criterion(pred_coords, pred_logits, coords, labels)

            if train:
                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(losses["total"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

            total_loss    += losses["total"].item()
            kp_loss_sum   += losses["kp_loss"].item()
            cls_loss_sum  += losses["cls_loss"].item()
            n_batches     += 1

            accumulator.update(pred_coords, coords, pred_logits, labels)

        metrics = accumulator.compute()
        metrics["loss"]      = total_loss   / n_batches
        metrics["kp_loss"]   = kp_loss_sum  / n_batches
        metrics["cls_loss"]  = cls_loss_sum / n_batches
        return metrics

    # ------------------------------------------------------------------

    def train(self):
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(self.train_loader, train=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, train=False)

            self.scheduler.step()
            lr = self.scheduler.get_lr()

            # Logging
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | lr={lr:.2e} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_pck25={val_metrics['pck25']:.4f} | "
                f"val_f1={val_metrics['macro_f1']:.4f} | "
                f"val_dist={val_metrics['mean_dist_px']:.1f}px | "
                f"{elapsed:.0f}s"
            )

            # Track history
            for k, v in val_metrics.items():
                self.history.setdefault(f"val_{k}", []).append(v)
            for k, v in train_metrics.items():
                self.history.setdefault(f"train_{k}", []).append(v)

            # Checkpoint: best PCK25
            current = val_metrics[self.monitor_metric]
            if current > self.best_metric:
                self.best_metric = current
                self.patience_counter = 0
                self._save("best_pck.pth", val_metrics, epoch)
                logger.info(f"  ↑ New best {self.monitor_metric}={current:.4f} — saved")
            else:
                self.patience_counter += 1

            # Checkpoint: best F1 (separate)
            if epoch == 1 or val_metrics["macro_f1"] >= max(
                self.history.get("val_macro_f1", [0.0])
            ):
                self._save("best_f1.pth", val_metrics, epoch)

            # Always save last
            self._save("last.pth", val_metrics, epoch)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        logger.info(f"Training done. Best {self.monitor_metric}={self.best_metric:.4f}")

    # ------------------------------------------------------------------

    def _save(self, filename: str, metrics: dict, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            self.output_dir / filename,
        )

#!/usr/bin/env python3
"""
train.py – Entry point for GCP model training.

Trains a single multi-task model on a MIX of crop scales (full image +
medium + tight crops, see configs/default.yaml `data.scale_choices`), so
the resulting checkpoint is directly usable for coarse-to-fine cascade
inference (scripts/inference.py) or even single-pass full-image inference.

Usage:
    python scripts/train.py \
        --data_root /path/to/train_dataset \
        --labels_json /path/to/curated_gcp_marks.json \
        --output_dir ./runs/exp1 \
        [--config configs/default.yaml] \
        [--resume ./runs/exp1/last.pth]
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow src/ imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.dataset import (
    GCPDataset,
    build_weighted_sampler,
    load_labels,
    train_val_split,
)
from src.model import build_model, load_checkpoint
from src.trainer import Trainer
from src.utils import get_device, load_config, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train GCP pose estimation model")
    p.add_argument("--data_root",    required=True, help="Path to train dataset directory")
    p.add_argument("--labels_json",  required=True, help="Path to curated_gcp_marks.json")
    p.add_argument("--output_dir",   default="./runs/exp1", help="Directory for checkpoints and logs")
    p.add_argument("--config",       default="configs/default.yaml", help="YAML config path")
    p.add_argument("--resume",       default=None, help="Path to checkpoint to resume from")
    p.add_argument("--freeze_epochs", type=int, default=0, help="Epochs to freeze backbone (0=no freeze)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)

    # Load config
    cfg = load_config(args.config)
    seed_everything(cfg["training"].get("seed", 42))
    device = get_device()
    logger.info(f"Device: {device}")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    all_labels = load_labels(args.labels_json)
    logger.info(f"Loaded {len(all_labels)} cleaned labels (normalization + manual fixes applied)")

    train_labels, val_labels = train_val_split(
        all_labels,
        val_ratio=cfg["training"].get("val_split", 0.15),
        seed=cfg["training"].get("seed", 42),
    )
    logger.info(f"Train: {len(train_labels)} | Val: {len(val_labels)}")

    img_size      = cfg["model"].get("img_size", 512)
    scale_choices = cfg["data"].get("scale_choices", [0, 384, 768, 1536])
    crop_jitter   = cfg["data"].get("crop_jitter", 200)
    val_scale     = cfg["data"].get("val_scale", 1024)

    logger.info(f"Train scale_choices={scale_choices} crop_jitter={crop_jitter}")
    logger.info(f"Val fixed scale={val_scale} (matches Stage-2 refine window)")

    train_dataset = GCPDataset(
        root=args.data_root,
        labels=train_labels,
        img_size=img_size,
        scale_choices=scale_choices,
        crop_jitter=crop_jitter,
        augment=True,
        aug_cfg=cfg.get("augmentation", {}),
    )
    val_dataset = GCPDataset(
        root=args.data_root,
        labels=val_labels,
        img_size=img_size,
        scale_choices=[val_scale],
        crop_jitter=0,
        augment=False,
    )

    sampler = build_weighted_sampler(train_dataset)
    bs = cfg["training"].get("batch_size", 16)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        sampler=sampler,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = build_model(cfg["model"], device)

    if args.resume:
        model = load_checkpoint(model, args.resume, device)
        logger.info(f"Resumed from {args.resume}")

    if args.freeze_epochs > 0:
        logger.info(f"Freezing backbone for {args.freeze_epochs} warmup epochs")
        model.freeze_backbone()

    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M"
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=args.output_dir,
        device=device,
    )

    # Handle backbone freeze/unfreeze
    original_train = trainer.train

    def train_with_unfreeze():
        for epoch_offset in range(args.freeze_epochs):
            trainer._run_epoch(train_loader, train=True)
            trainer.scheduler.step()
        if args.freeze_epochs > 0:
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen")
        original_train()

    if args.freeze_epochs > 0:
        trainer.train = train_with_unfreeze

    trainer.train()


if __name__ == "__main__":
    main()

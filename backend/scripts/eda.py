#!/usr/bin/env python3
"""
eda.py – Exploratory Data Analysis for the GCP dataset.

Uses `src.dataset.load_labels`, which already applies:
  - shape-string normalization (e.g. "L-Shape" -> "L-Shaped")
  - the 4 manually-reviewed missing-shape labels (MANUAL_SHAPE_LABELS)

so the statistics below reflect the CLEANED label set used for training.

Prints and saves statistics about:
  - Class distribution (shape counts)
  - Keypoint coordinate distribution (x, y histograms)
  - Image resolution distribution
  - Keypoint distance from image center
  - Potential data quality issues (near-edge keypoints, duplicates)

Usage:
    python scripts/eda.py \
        --data_root /path/to/train_dataset \
        --labels_json /path/to/curated_gcp_marks.json \
        [--output_dir ./eda_output]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--labels_json", required=True)
    p.add_argument("--output_dir", default="./eda_output")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Cleaned labels: normalization + manual missing-shape fixes applied.
    labels = load_labels(args.labels_json)

    print(f"\n{'='*60}")
    print(f"  GCP Dataset EDA — {len(labels)} samples (after cleaning)")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # 1. Class distribution
    # ---------------------------------------------------------------
    shapes = [v["verified_shape"] for v in labels.values()]
    shape_counts = Counter(shapes)
    print("\n[1] Shape class distribution:")
    total = len(shapes)
    for cls, cnt in sorted(shape_counts.items()):
        print(f"    {cls:12s}: {cnt:4d}  ({100*cnt/total:.1f}%)")

    # ---------------------------------------------------------------
    # 2. Coordinate statistics
    # ---------------------------------------------------------------
    xs = [v["mark"]["x"] for v in labels.values()]
    ys = [v["mark"]["y"] for v in labels.values()]

    print("\n[2] Keypoint coordinate statistics:")
    print(f"    x  →  min={min(xs):.1f}  max={max(xs):.1f}  "
          f"mean={sum(xs)/len(xs):.1f}  std={_std(xs):.1f}")
    print(f"    y  →  min={min(ys):.1f}  max={max(ys):.1f}  "
          f"mean={sum(ys)/len(ys):.1f}  std={_std(ys):.1f}")

    # ---------------------------------------------------------------
    # 3. Image resolution + proximity to edges
    # ---------------------------------------------------------------
    root = Path(args.data_root)
    near_edge = []
    resolutions = []
    edge_thresh = 50  # pixels

    print("\n[3] Checking image files...")
    missing = 0
    for rel_path, ann in labels.items():
        img_path = root / rel_path
        if not img_path.exists():
            missing += 1
            continue
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                W, H = img.size
                resolutions.append((W, H))
            x, y = ann["mark"]["x"], ann["mark"]["y"]
            if x < edge_thresh or x > W - edge_thresh or y < edge_thresh or y > H - edge_thresh:
                near_edge.append(rel_path)
        except Exception as e:
            print(f"    [WARN] Could not open {img_path}: {e}")

    if missing:
        print(f"    [WARN] {missing} images listed in JSON but NOT FOUND on disk")

    if resolutions:
        ws = [r[0] for r in resolutions]
        hs = [r[1] for r in resolutions]
        print(f"    Width  → min={min(ws)} max={max(ws)} mean={sum(ws)/len(ws):.0f}")
        print(f"    Height → min={min(hs)} max={max(hs)} mean={sum(hs)/len(hs):.0f}")
        unique_res = Counter(resolutions)
        print(f"    Unique resolutions: {len(unique_res)}")
        for res, cnt in unique_res.most_common(5):
            print(f"      {res[0]}x{res[1]}: {cnt} images")

    print(f"\n    Near-edge keypoints (<{edge_thresh}px from border): {len(near_edge)}")
    if near_edge[:5]:
        for path in near_edge[:5]:
            print(f"      {path}")

    # ---------------------------------------------------------------
    # 4. Per-project/survey breakdown
    # ---------------------------------------------------------------
    projects = Counter(Path(p).parts[0] for p in labels.keys())
    print(f"\n[4] Project breakdown ({len(projects)} projects):")
    for proj, cnt in projects.most_common(10):
        print(f"    {proj:30s}: {cnt} images")

    # ---------------------------------------------------------------
    # 5. Potential duplicates (same x,y coords)
    # ---------------------------------------------------------------
    coord_set = Counter((round(v["mark"]["x"], 1), round(v["mark"]["y"], 1))
                        for v in labels.values())
    dups = {k: v for k, v in coord_set.items() if v > 1}
    print(f"\n[5] Duplicate coordinates (same x,y): {len(dups)} groups")
    if dups:
        for coord, cnt in list(dups.items())[:5]:
            print(f"    {coord}: {cnt} images")

    # ---------------------------------------------------------------
    # Save summary + cleaned labels
    # ---------------------------------------------------------------
    summary = {
        "n_samples": total,
        "shape_counts": dict(shape_counts),
        "x_stats": {"min": min(xs), "max": max(xs), "mean": sum(xs)/len(xs)},
        "y_stats": {"min": min(ys), "max": max(ys), "mean": sum(ys)/len(ys)},
        "near_edge_count": len(near_edge),
        "missing_files": missing,
    }
    with open(out / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "clean_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSummary saved to {out / 'eda_summary.json'}")
    print(f"Cleaned labels saved to {out / 'clean_labels.json'}")
    print(f"{'='*60}\n")


def _std(vals):
    mu = sum(vals) / len(vals)
    return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5


if __name__ == "__main__":
    main()

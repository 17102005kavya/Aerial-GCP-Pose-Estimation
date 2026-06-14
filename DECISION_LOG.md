# Engineering Decision Log

## Project
Aerial GCP Localization & Shape Classification Platform

**Architecture:** EfficientNet-B3 Multi-Scale Cascade  
**Deployment:** FastAPI + PyTorch Backend, Next.js 15 Dashboard

---

## Decision 1: EfficientNet-B3 Backbone

### Problem
Ground Control Point (GCP) markers occupy a very small fraction of high-resolution aerial images, requiring a model that can capture fine geometric features while remaining efficient enough for repeated inference.

### Alternatives Considered
- ResNet-50
- ConvNeXt
- EfficientNet-B3

### Decision
Selected **EfficientNet-B3** pretrained on ImageNet.

### Why?
- Strong accuracy-to-compute ratio
- Efficient for multi-stage inference
- Effective at capturing edge and corner features critical for GCP localization

### Tradeoff
Lower representational capacity than larger ConvNeXt variants.

---

## Decision 2: Spatial Attention Pooling

### Problem
Global Average Pooling removes spatial information needed for coordinate regression.

### Alternatives Considered
- Global Average Pooling
- Flattening the feature map
- Spatial Attention Pooling

### Decision
Implemented **Spatial Attention Pooling**.

### Why?
Allows the network to learn which spatial regions are important before pooling.

### Tradeoff
Slight increase in parameters and computation.

---

## Decision 3: Multi-Scale Training

### Problem
Models trained only on tight crops fail when presented with full-resolution aerial images.

### Alternatives Considered
- Separate detector and refiner models
- Single multi-scale model

### Decision
Train a single model on multiple crop scales:

```text
[Full Image, 1536px, 768px, 384px]
```

### Why?
A single checkpoint can localize markers at every zoom level encountered during inference.

### Tradeoff
The model must learn both coarse and fine localization simultaneously.

---

## Decision 4: Crop Jitter Augmentation

### Problem
Perfectly centered training crops cause the model to assume the marker is always centered.

### Decision
Apply random crop offsets up to ±200 pixels.

### Why?
Improves robustness during cascade refinement stages.

---

## Decision 5: Fixed Validation Scale

### Problem
Multi-scale validation produces unstable and incomparable metrics.

### Decision
Validate using a fixed 1024px crop.

### Why?
Provides a stable benchmark for checkpoint selection and early stopping.

---

## Decision 6: Coarse-to-Fine Cascade Inference

### Problem
Single-pass regression struggles to achieve high localization precision on 4000×3000 images.

### Alternatives Considered
- Single-pass regression
- Two-stage refinement
- Multi-stage cascade

### Decision
Use a four-stage cascade:

```text
Full Image
    ↓
1536px Crop
    ↓
768px Crop
    ↓
384px Crop
```

### Why?
Each stage progressively refines the prediction using a smaller search region.

### Tradeoff
Large errors at early stages may propagate.

---

## Decision 7: Test-Time Augmentation (TTA)

### Problem
Predictions can vary with image orientation.

### Decision
Apply 4-way flip TTA during inference.

### Why?
Reduces variance and improves localization stability.

---

## Decision 8: Multi-Task Learning

### Problem
The model must predict both coordinates and shape classes.

### Decision
Use a dual-head architecture:

- Regression Head
- Classification Head

### Loss Function

```text
Total Loss =
Wing Loss +
0.25 × Cross Entropy Loss
```

### Why?
Localization is prioritized while maintaining shape classification performance.

---

## Decision 9: Handling Class Imbalance

### Problem
L-Shaped markers are significantly underrepresented.

### Decision
Use a Weighted Random Sampler.

### Why?
Ensures minority classes appear more frequently during training.

---

## Decision 10: Label Normalization

### Problem
Annotations contained inconsistent label names.

Examples:

```text
L Shape
L-Shape
l-shaped
Squares
cross
```

### Decision
Normalize all labels to:

```text
Cross
L-Shaped
Square
```

### Why?
Prevents label fragmentation and training inconsistencies.

---

## Decision 11: Missing Annotation Recovery

### Problem
Several samples were missing valid shape labels.

### Decision
Manually review affected images and assign verified labels.

### Why?
Preserves valuable training data rather than discarding samples.

---

# Results

| Metric | Result |
|----------|----------|
| Validation PCK@25 | 0.9867 |
| Mean Localization Error | ~7–9 px |
| Validation F1 Score | 1.0000 |
| Backbone | EfficientNet-B3 |
| Input Resolution | 512×512 |
| Inference Pipeline | 4-Stage Cascade |

---

# Future Work

1. Evaluate performance on the hidden test set.
2. Audit Stage-0 predictions separately.
3. Verify validation class balance for minority classes.
4. Explore larger backbones if tighter localization accuracy is required.
5. Improve model versioning and deployment workflows.

---

# Model Weights

The trained model weights are hosted separately due to file size constraints.

Download:

https://drive.google.com/file/d/1foVQ4ED3TafLvuFyQrbcC9fcxW341nUv/view?usp=sharing

Place the downloaded checkpoint in:

```text
models/
└── best_pck.pth
```

---

# Sample Predictions

Example predictions generated on the test dataset are available in:

```text
sample_predictions/
└── predictions.json
```

These files demonstrate the expected output format produced by the inference pipeline.

---

# Running Inference

```bash
python scripts/inference.py \
    --data_root <TEST_DATA_PATH> \
    --checkpoint models/best_pck.pth \
    --output predictions.json \
    --config configs/default.yaml \
    --tta
```

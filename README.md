# inat_phenology_cv
Phenology vision model enrichment for [inat-obs-scorer](https://github.com/etiennegodin/inat-obs-scorer)

## v0.2
- Observation-level mean pooling
- Observation-aware train/val/test split to prevent leakage
- ~5.5k observations with ~14.5k images
    - ~50/50 class balance
    - Mean of 2.65 img per obs, with std of ~2.15
    - Median of 2 img per obs
- Partial backbone unfreeze (`blocks.6`, `conv_head`, `bn2`) with per-layer LRs (backbone: 1e-5, head: 1e-3)

- All observations in a batch are `torch.cat` into one stacked tensor → single backbone forward pass → `torch.split` recovers per-observation chunks → `.mean(0)` pools each observation → head classifies
    - Key insight: stacking is critical for BatchNorm stability — per-observation loops starved BatchNorm with 2-3 samples, collapsing training signal
- Final results: val loss ~0.313, ROC ~0.936 with ~14k images

### Dataset distribution


| Classes | n | images | mean_img_per_obs
|---|---|---|---|
flowering	| 2689 | 	7304 | 	2.72 |
non_flowering | 	2766| 	7130| 	2.58|

## Roadmap

### ✅ v0.1 — Baseline
- Transfer learning from ResNet-50 frozen backbone
- Custom head: Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→1)
- Single photo per observations
- BCEWithLogitsLoss, Adam
- 736 images, 60/20/20 stratified split, early stopping (patience=3), best checkpoint saving
- evaluate() returns loss, accuracy, confusion matrix, ROC-AUC
- ~80% val accuracy, ~0.90 roc_auc

### ✅ v0.2 — Multi-image classification

- Backbone swap
- Data expansion, multiple photos per observations
- Level 1 multi-image: weak supervision baseline
- Level 2 multi-image: observation-level mean pooling
- MlFlow tracking setup

### ✅  v0.3 — Multiple Instance Learning

- Attention-based MIL

### 🔲 v0.4 — Multi-label classification

- Multi-label classification expanding to full Flowers+Fruits phenology

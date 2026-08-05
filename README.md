# inat_phenology_cv
Phenology vision model enrichment for [inat-obs-scorer](https://github.com/etiennegodin/inat-obs-scorer)

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

### ✅ v0.4 — Multi-label classification

- Model improvements
    - Multi-label classification expanding to full Flowers+Fruits phenology
    - Learning rate scheduling
- Pipeline features
    - Multiple backbone compatibility
        - EfficientNetB0
        - BioCLIP

### 🔲 v0.5 — Optuna set up
- Optuna setup

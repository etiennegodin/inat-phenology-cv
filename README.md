# inat_phenology_cv
Vision model enrichment source for inat-obs-scorer

## Roadmap

### ✅ v0.1 — Baseline
- Transfer learning from ResNet-50 frozen backbone
- Custom head: Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→1)
- Single photo per observations
- BCEWithLogitsLoss, Adam
- 3473 images, 60/20/20 stratified split, early stopping (patience=3), best checkpoint saving
- evaluate() returns loss, accuracy, confusion matrix, ROC-AUC
- ~826% val accuracy, ~0.914 roc_auc

### 🔲 v0.2 — Multi-label classification

- Backbone swap
- Data expansion, multiple photos per observations
- Level 1 multi-image: weak supervision baseline
- Level 2 multi-image: observation-level mean pooling
- MlFlow and optuna setup

### 🔲 v0.3 — Multiple Instance Learning

- Multi-label classification expanding to full Flowers+Fruits phenology
- Attention-based MIL

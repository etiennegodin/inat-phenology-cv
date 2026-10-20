# Project history

The project began as an enrichment step for a companion project, [inat-obs-scorer](https://github.com/etiennegodin/inat-obs-scorer), and grew from there. Each version is a response to a limit I hit in the one before.

| Version | What changed | Why |
|---|---|---|
| **v0.1** Baseline | Frozen ResNet-50, custom head (2048 → 256 → ReLU → Dropout → 1), one photo per observation, BCE loss. 736 images, stratified split, early stopping. About 80% validation accuracy and 0.90 ROC-AUC. | Prove the idea works at all. |
| **v0.2** Multi-image | Swapped backbones, expanded the data, trained on multiple photos per observation (first with weak supervision, then mean pooling). MLflow tracking added. | One photo often misses the informative one. |
| **v0.3** MIL | Attention-based multiple-instance learning. | Mean pooling treats every photo equally. |
| **v0.4** Multi-label | Flowers and fruits as separate classes, learning-rate scheduling, multiple backbones (EfficientNet-B0, BioCLIP). | An observation can be several things at once. |
| **v0.5** Pipeline | Better step and metric logging, Colab compatibility, a centralized MLflow server. | I was running on more than one machine. |
| **v0.6** CLAM-MB | Multi-branch attention, reworked attention metrics, per-class loss logging. | One attention head for all classes was a bottleneck. |
| **v0.7** Hardware | `HardwareProfile`, dataset caching decisions, bf16/fp16 support, v2 transforms that accept cached tensors. | Training had to work on both a tiny local GPU and Colab. |
| **v0.8** Objectives | Per-class patience counters, a metrics overhaul (compute, log and plot separated), gated attention with dropout, seeding for determinism. | Early stopping and metrics were misleading me. |
| **v0.9** Unfreezing *(in progress)* | Progressive backbone unfreezing, Cleanlab audits of validation errors. | Test whether fine-tuning deeper blocks helps. |
| **v1.0** Tuning *(in progress)* | Optuna search, two-stage design. | Stop hand-tuning. |

<!-- TODO: confirm v0.9/v1.0 status, and decide whether to add rows for the backbone × attention comparison and the serving layer (pyfunc registration, DuckDB prediction store). -->

## Where it's going

- A controlled comparison of backbones and attention architectures
- A settled rule for when unfreezing fires
- A Cleanlab audit of the training set using a k-fold proxy
- A finished serving path: registered model, validated scoring, prediction store
- Possibly publishing model weights on Hugging Face

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
| **v0.9** Inference | MLflow pyfunc registered model, and a command that turns an iNaturalist URL into a prediction. | A model that can only be evaluated inside the training code isn't much use. |
| **v0.10** Backbones and aggregation | DINOv3 backbone and a cross-attention aggregation build, alongside CLAM-MB. | Compare backbones and aggregation strategies under one config. |
| **v0.11** Label audit *(in progress)* | Cleanlab k-fold audit of the training labels. | Tune and unfreeze on cleaner labels. |
| **v0.12** Unfreezing *(planned, mechanism built)* | Progressive backbone unfreezing, validated with a controlled experiment. | Test whether fine-tuning deeper blocks helps. |
| **v1.0** Tuning *(planned)* | Optuna search, two-stage design. | Stop hand-tuning. |

<!-- TODO: version numbers from v0.9 on are my guess. Adjust them to match your own roadmap. -->

## Where it's going

- A controlled comparison of backbones and aggregation strategies
- Finish the Cleanlab audit and act on what it finds
- A settled rule for when unfreezing fires, then the tuning rounds
- Hosting inference, possibly behind a small API
- Possibly publishing model weights on Hugging Face

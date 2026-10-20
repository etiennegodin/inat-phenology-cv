# Architecture

The system keeps model construction, experiment control and data operations apart, so I can swap a backbone or an attention variant without rewriting the training loop.

## Package layout

Everything lives under `src/plant_pheno/`:

| Module | Responsibility |
|---|---|
| `cli.py` | `torch_pipe` entry point |
| `config/` | Hydra composition, registry-backed choices, search spaces, runtime and system resolution |
| `core/` | Backbone integration, the MIL model, device handling, and the MLflow pyfunc wrapper |
| `train/` | Dataset, batch sampler, dataloader, workflow and runner, metrics, flow control (stopping and unfreezing), persistence |
| `tune/` | Optuna objective |
| `inference/` | Photo sync, record validation, predictor, scoring pipeline |
| `inat_client/` | iNaturalist API access, split into clients, fetchers and writers |
| `data/` | DuckDB access, SQL and adapters |
| `infra/` | Logging, seeding, MLflow helpers, networking |
| `cleanlab.py`, `review.py`, `register.py`, `val.py`, `status.py` | Command implementations behind the CLI |

## Two install profiles

This codebase holds both the training pipeline and the inference pipeline, but they don't need the same dependencies. A plain `pip install` gives the inference side: load a registered model and score observations. The full training stack (training, tuning, label audits) needs the `[train]` extra. Keeping them apart means someone who only wants predictions doesn't have to install the training dependencies.

## From config to a run

Hydra picks four things: the environment (`env`: local or Colab), the action (`command`), the model variant (`architecture`) and optional presets (`experiment`). A run then goes through these steps:

1. Compose the config and resolve runtime settings and paths.
2. Detect hardware and decide on dataset caching.
3. Build the dataset and dataloaders.
4. Build the backbone and the attention head.
5. Train with the configured optimizer, metrics and stopping rules.
6. Log metrics and artifacts to MLflow, and save checkpoints.

Having one config path for everything is what lets me compare runs fairly. Fewer one-off scripts means fewer hidden differences between experiments.

## The model

Each photo goes through a pretrained backbone. I've used BioCLIP ViT-B/16, BioCLIP 2 and DINOv3, and the backbone is swappable through config. The photo embeddings are then combined into one prediction per class by one of two aggregation strategies:

- **MIL (CLAM-MB):** the `mil_pooling` architecture. It started as plain attention-based MIL and grew into the multi-branch design below.
- **Cross-attention:** the `cross_attention` architecture, a newer alternative added alongside the DINOv3 backbone.

The details below describe the MIL (CLAM-MB) model.

- **Multi-branch attention (CLAM-MB style):** every class gets its own attention network and classifier head, so each class can focus on different photos. All branches reuse one backbone forward pass.
- **Per-class sigmoid outputs** handle the multi-label setup.
- **Gated attention** and **attention dropout** are configurable.
- **L2 normalization** of the embeddings happens before attention pooling. CLIP-style pretraining only ever trains on normalized vectors, so the raw output magnitude is unconstrained and unreliable.
- **Variable-length bags:** the model takes a list of per-observation tensors. Photos are stacked within an observation, never across observations.
- **Memory:** `max_images`, the total number of photos per forward pass, is what actually drives VRAM, not the observation batch size.

Because both aggregation strategies and several backbones sit behind config, the backbone × aggregation grid is something I can run as a controlled comparison instead of a pile of one-off scripts.

## Hardware and caching

A `HardwareProfile` reads the machine's resources. A resolver turns it, together with the dataset's memory footprint, into a cache decision, and the dataset is built to match. That's how the same code runs on a 2 GB local GPU and on a Colab T4 or L4. Mixed precision (bf16 and fp16) is supported, and images stay as uint8 until normalization to keep the cache small.

## Data layer

Data lives in DuckDB, with a raw and staged pattern. Training and serving databases are separate, so serving can deploy without the training data. Label corrections are an overlay file keyed by observation ID, not in-place edits, so the original labels stay intact.

## Separation of concerns

- Architecture lives in the model classes.
- Training strategy (freezing, optimizer parameter groups) lives in the training workflow.
- Config knows about environments, and nothing else needs to.
- Inference is a separate layer, so data sync never gets tangled into the training loop. See [inference and deployment](inference-and-deployment.md).

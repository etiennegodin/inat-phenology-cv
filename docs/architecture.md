# Architecture and design

## Package structure

The main implementation sits under `src/plant_pheno` and is structured around functional layers:

- `config/`: Hydra config, runtime resolution, registry-backed choices, and validation
- `train/`: training loop, metrics, checkpointing, flow control, and data pipeline construction
- `inference/`: observation syncing, validation, predictor wiring, and prediction logging
- `tune/`: Optuna objective generation and tuning configuration
- `core/`: model-building logic and backbone model integration
- `data/`: dataset construction, persistence, and model evaluation utilities
- `infra/`: logging, seeding, MLflow URI resolution, and environment utilities
- `utils/`: shared helpers and class/object instantiation utilities

## Training pipeline

At a high level, the pipeline is:

1. Load Hydra configuration.
2. Resolve runtime settings and paths.
3. Build the dataset and dataloaders.
4. Instantiate the selected backbone and prediction head.
5. Run training with metrics, checkpointing, and optimization.
6. Log the run to MLflow with metrics and artifacts.
7. Save checkpoints and evaluation outputs.

The training stack supports multiple backbone families and architecture variants, while keeping model configuration centralized in the config system.

## Multi-image / MIL thinking

The project does not treat each photo as a standalone sample. Instead, a single observation can carry multiple photos, and the model aggregates the image-level evidence into an observation-level prediction.

The model stack supports:

- aggregated observation pooling
- attention-based weighting over image instances
- class-specific logging and metrics
- per-class loss and objective tracking

## Configuration model

The configuration layer is central to how the project is run:

- `env` controls environment assumptions and runtime setup.
- `command` selects the action: train, tune, val, cleanlab, status, etc.
- `architecture` selects model or pipeline variants.
- `experiment` can enable special training behaviors such as progressive unfreezing.

This ensures that feature selection and runtime behavior are not scattered through ad hoc scripts. The project prefers structured composition over flag-heavy manual branching.

## Experimentation and evaluation

The project includes:

- epoch-level metrics and class-specific logs
- validation runs with thresholded and raw predictions
- attention metrics summarizing per-observation relevance
- adaptive stopping and checkpoint objectives
- dataset and hardware-aware runtime decisions

This is designed for fast iteration on model quality, not just one-off training scripts.

## Inference pipeline

Inference is a separate layer from training. The system can:

- fetch observation records and photo URLs
- validate the resulting record schema
- sync local photo assets
- run a registered model through the predictor
- log results to the local DuckDB-backed store

This makes the project useful for both experimental training and operational scoring workflows.

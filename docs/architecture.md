# Architecture and design

The system separates model construction, experiment control, and data operations so that architecture choices can change without rewriting the surrounding workflow.

## Package structure

The implementation under `src/plant_pheno` is organized by responsibility:

- `config/`: Hydra config, runtime resolution, registry-backed choices, and validation
- `train/`: training loop, metrics, checkpointing, flow control, and data pipeline construction
- `inference/`: observation syncing, validation, predictor wiring, and prediction logging
- `tune/`: Optuna objective generation and tuning configuration
- `core/`: model-building logic and backbone model integration
- `data/`: dataset construction, persistence, and model evaluation utilities
- `infra/`: logging, seeding, MLflow URI resolution, and environment utilities
- `utils/`: shared helpers and class/object instantiation utilities

## From configuration to model run

Hydra configuration selects the environment, command, architecture, and experiment settings. The training workflow resolves those settings, constructs the data pipeline and model, and manages optimization and evaluation:

1. Load Hydra configuration.
2. Resolve runtime settings and paths.
3. Build the dataset and dataloaders.
4. Instantiate the selected backbone and prediction head.
5. Train with configured optimization, metrics, and checkpoint objectives.
6. Record run metrics and artifacts in MLflow.
7. Persist checkpoints and evaluation outputs for later inspection or use.

Centralizing model and runtime choices keeps experiments comparable and reduces the need for one-off training scripts.

## Multi-image / MIL thinking

An observation may contain multiple photos, so image-level evidence must be combined before producing the observation-level prediction. The model stack supports several aggregation strategies, including attention-based weighting of image instances.

The model stack supports:

- observation pooling and attention-based instance weighting
- class-specific metrics and loss aggregation
- per-class objective and training-state tracking

## Configuration model

The configuration layer is central to how the project is run:

- `env` controls environment assumptions and runtime setup.
- `command` selects the action: train, tune, val, cleanlab, status, etc.
- `architecture` selects model or pipeline variants.
- `experiment` can enable special training behaviors such as progressive unfreezing.

This keeps model and workflow choices explicit while allowing focused command-line overrides. See [the configuration guide](../configs/README.md) for composition examples.

## Experimentation and evaluation

The experiment workflow includes:

- epoch-level metrics and class-specific logs
- validation runs with thresholded and raw predictions
- attention metrics summarizing per-observation relevance
- adaptive stopping and checkpoint objectives
- dataset and hardware-aware runtime decisions

Together, these controls make it possible to compare experiments and inspect more than a single aggregate validation score.

## Inference pipeline

Inference is a separate layer from training. The system can:

- fetch observation records and photo URLs
- validate the resulting record schema
- sync local photo assets
- run a registered model through the predictor
- log results to the local DuckDB-backed store

The separate inference layer gives model experiments a path into repeatable batch scoring without coupling data synchronization to the training loop.

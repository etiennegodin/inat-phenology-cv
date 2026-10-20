# Plant Phenology CV

An end-to-end computer-vision system for recognizing plant phenology from iNaturalist observations. It brings together multi-image, multi-label modeling and the engineering around it: data preparation, controlled experiments, evaluation, and repeatable batch inference.

Plant observations often contain several photos, uneven visual evidence, and more than one active phenology label. This project treats the observation, rather than an individual image, as the prediction unit and provides a configurable workflow for experimenting with that problem.

## What the system does

The workflow starts with observation records and photo URLs, prepares image bags, and trains configurable models to predict multiple phenology classes. Attention-based multiple-instance learning can weight image evidence within an observation. Training runs log metrics and artifacts to MLflow, while Optuna supports parameter search and staged unfreezing provides a way to fine-tune deeper backbone blocks.

For inference, the pipeline validates records, synchronizes the required photos, loads a tracked model, and stores structured predictions in DuckDB. Keeping training and inference in one package makes experiments easier to carry through to repeatable scoring workflows.

## Engineering highlights

- Observation-level modeling for multi-photo, multi-label ecological data
- Pluggable backbone and architecture choices, composed through Hydra
- Experiment tracking, checkpointing, evaluation, and Optuna tuning
- Progressive backbone unfreezing with class-aware training controls
- Label-quality review support with CleanLab
- Validated batch inference with local photo synchronization and DuckDB persistence
- CLI workflows for training, tuning, validation, and status checks

The project is actively evolving as a research and engineering system. The repository focuses on making experiments configurable and traceable, while keeping the path from model development to inference explicit.

## Core commands

The CLI entry point is `torch_pipe`:

Examples:

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status
```

Hydra composes runtime, command, and architecture choices, with individual settings available as overrides. See [the configuration guide](configs/README.md) for examples.

## Repository structure

- `src/plant_pheno/` — package implementation
- `src/plant_pheno/core/` — model construction and backbone integration
- `src/plant_pheno/train/` — datasets, training workflow, metrics, and checkpoints
- `src/plant_pheno/inference/` — record validation, photo sync, and prediction
- `src/plant_pheno/tune/` — Optuna objective and search-space integration
- `src/plant_pheno/config/` and `configs/` — Hydra composition and runtime settings
- `tests/` — regression coverage for models, metrics, data quality, and training control

## Documentation map

Explore the system by topic:

- [docs/overview.md](docs/overview.md) — problem framing and project goals
- [docs/architecture.md](docs/architecture.md) — model design and system components
- [docs/training-and-tuning.md](docs/training-and-tuning.md) — training workflow and experiment strategy
- [docs/inference-and-deployment.md](docs/inference-and-deployment.md) — batch scoring and persistence

# Plant Phenology CV

This project builds a computer-vision pipeline for phenology classification from iNaturalist-style observation data. The system trains multi-image, multi-label models for plant phenology events and exposes a reusable inference pipeline that syncs photos, validates observations, runs model inference, and writes results back to the local data store.

The repository is no longer a single-model prototype. It is a feature-complete training and inference stack built around Hydra configuration, MLflow tracking, dynamic backbones, Optuna tuning, attention-based multiple-instance learning, and progressive backbone unfreezing.

## Current project scope

The code currently supports:

- Multi-label phenology prediction for observation-level labels
- Multiple photos per observation using bag-based training patterns
- Attention-based MIL and pooled observation-level inference
- Multiple backbone options and architecture variants
- Hydra-driven configuration and validation
- MLflow experiment tracking and artifact logging
- Optuna search-space tuning for training and model parameters
- Progressive unfreezing of backbone blocks
- CleanLab-based label-quality checks for training/validation refinement
- DuckDB-backed persistence and local data synchronization for inference
- CLI-driven workflows for train, tune, validate, and status checks

## High-level architecture

The project is organized as a modular ML pipeline:

- Data ingestion and validation from iNaturalist records and photo URLs
- Dataset construction and batching for multi-image observations
- Model construction from Hydra configuration, including backbone selection and training heads
- Training workflow with metrics, early stopping, checkpointing, and unfreezing logic
- Model evaluation and artifact logging via MLflow
- Inference pipeline that fetches the required images, validates records, and logs predictions

## Core commands

The primary CLI entry point is `torch_pipe`.

Examples:

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status
```

The configuration system is Hydra-based and supports composition via `env`, `command`, and `architecture` groups. See the configuration docs for the current command and experiment patterns.

## Repository structure

- `src/plant_pheno/` — main package code
- `src/plant_pheno/train/` — training workflow, metrics, checkpointing, and flow control
- `src/plant_pheno/inference/` — sync + validation + prediction pipeline
- `src/plant_pheno/tune/` — Optuna objective and search-space wiring
- `src/plant_pheno/config/` — Hydra config, validation, and runtime resolution
- `tests/` — regression tests covering backbone behavior, metrics, determinism, cleanlab, and unfreezing flow
- `configs/` — config defaults, Hydra examples, and command patterns
- `notes/` — design and refactor notes capturing the evolution of the pipeline

## What changed from the older README

The original roadmap described a simple baseline and a few later milestones. The current codebase has moved well beyond that:

- The project is now configured as a real package named `plant_pheno`
- Training is Hydra-driven, not ad hoc notebook code
- Multi-label training and attention-based observation pooling are part of the core workflow
- MLflow is integrated as the experiment and registry layer
- Optuna is a first-class tune command, not a future idea
- Progressive unfreezing and per-class patience logic are implemented and tested
- CleanLab, validation, and inference pipelines are active pieces of the system
- DuckDB-backed data operations and model inference are part of the production workflow

## Documentation map

This README is intentionally a landing page. The detailed project context is split into focused docs:

- [docs/overview.md](docs/overview.md) — project mission, scope, and current status
- [docs/architecture.md](docs/architecture.md) — model architecture and pipeline components
- [docs/training-and-tuning.md](docs/training-and-tuning.md) — training workflow, CLI commands, and experimentation
- [docs/inference-and-deployment.md](docs/inference-and-deployment.md) — sync, validation, model serving, and result logging

## Current development status

This repo is in active research-and-operations development. The codebase is structured for iterative experimentation, model comparison, and regular retraining, rather than a static demo implementation.

If you are onboarding to the project, start with the docs listed above before diving into the training and inference modules.

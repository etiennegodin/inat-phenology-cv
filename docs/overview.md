# Project overview

## Mission

This project creates a vision pipeline for plant phenology classification using observation-level imagery and metadata from iNaturalist-style datasets. The goal is to estimate phenology states from photo sets, not just single images, and to support this as a repeatable ML workflow with experiment tracking, evaluation, and model deployment.

## Current state

The codebase has evolved from a baseline transfer-learning prototype into a structured ML project with:

- multi-label phenology classification
- multi-image observation bags
- attention-based multiple-instance learning
- configurable backbone stacks
- Hydra-driven runtime configuration
- MLflow tracking and artifact logging
- Optuna-based hyperparameter tuning
- progressive unfreezing for backbone training
- CleanLab-driven label review support
- inference and validation pipelines tied to local data stores

## Why this exists

The project sits between data collection and operational ecological scoring. It turns raw observation metadata and images into model predictions that can later be used in broader phenology and biodiversity workflows. This is not only a model benchmark; it is an end-to-end system for experimentation and pipeline execution.

## Project goals

1. Train robust phenology models from noisy, multi-image ecological observations.
2. Support multiple backbone architectures and model families.
3. Keep the training stack reproducible and trackable via MLflow and structured config.
4. Provide an inference pipeline that can ingest observation URLs and synchronize needed photos.
5. Support iterative improvement through tuning, validation, and label-quality workflows.

## Core terms

- Observation: a single ecological sample with one or more photos and phenology metadata.
- Bag-of-images: the set of images attached to an observation used in a multi-instance prediction pipeline.
- Multi-label classification: each observation can have multiple active phenology classes, such as flowering and fruiting states.
- Progressive unfreezing: a staged backbone training approach that gradually makes deeper layers trainable.

## Repository direction

The repository is organized for active experimentation and repeatable training jobs rather than a single static demo. The current emphasis is on research-grade ML workflow structure: strong config management, reproducible experiment tracking, robust validation, and inference pipelines that can be rerun reliably.

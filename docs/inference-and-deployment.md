# Inference and deployment workflow

## Overview

The inference path is designed for operational scoring: given a set of observation URLs or records, the system fetches the required photos, validates the data, runs a model, and logs the predictions in a local data store.

This is a real workflow layer rather than a toy demo notebook. It is built to connect the training pipeline to downstream ecological analysis.

## Inference pipeline steps

The core inference flow is:

1. Fetch observation metadata and photo URLs.
2. Validate record structure and required fields.
3. Synchronize local image assets.
4. Load the registered model from MLflow.
5. Run model inference on the photo bag.
6. Convert outputs into structured prediction records.
7. Log outputs to DuckDB and related storage.

## Supporting components

The project includes:

- `InatDataSynchronizer` for preparing local photo and data state
- `PhenologyPredictor` for model loading and prediction execution
- `validate_records` / `records_to_dataframe` for record-level validation
- persisted model results and diagnostics through the local app data stack

## Deployment model

The overall deployment pattern is best described as a model-backed workflow pipeline rather than a single web service endpoint. The project is structured around:

- reproducible training runs
- MLflow model registration and retrieval
- data sync and validation before scoring
- local ingestion of prediction artifacts

This makes the project convenient for repeated batch scoring or research workflows, especially when the source data is observational and photo-heavy.

## Operational considerations

The project is designed with practical concerns in mind:

- environment-aware configuration
- photo synchronization and local cache usage
- model retrieval from tracked experiment artifacts
- logging of both predictions and metadata
- compatibility with Colab and local machine environments

## Current status

The repository should be understood as an active ML operating pipeline, not just a model training script. The inference layer is one of the major pieces that has grown beyond the original baseline roadmap and should be treated as part of the active system design.

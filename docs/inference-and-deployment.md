# Inference and deployment workflow

## Overview

The inference workflow carries a trained model into batch scoring. Given observation records or URLs, it validates the input, synchronizes the associated photos, runs prediction, and persists structured results for downstream analysis.

This keeps data preparation, model execution, and prediction storage in a single traceable path from input records to analysis-ready outputs.

## Inference pipeline steps

The scoring path is:

1. Fetch observation metadata and photo URLs.
2. Validate record structure and required fields.
3. Synchronize local image assets.
4. Load the selected model from MLflow.
5. Run model inference on the photo bag.
6. Convert outputs into structured prediction records.
7. Persist outputs to DuckDB and the surrounding local data store.

## Supporting components

The project includes:

- `InatDataSynchronizer` for preparing local photo and data state
- `PhenologyPredictor` for model loading and prediction execution
- `validate_records` / `records_to_dataframe` for record-level validation
- persisted model results and diagnostics through the local app data stack

## Deployment model

The current deployment pattern is a model-backed workflow rather than a continuously running web endpoint. It is organized around:

- reproducible training runs
- MLflow model registration and retrieval
- data sync and validation before scoring
- local ingestion of prediction artifacts

This pattern suits repeated research and batch-scoring workflows where source observations are photo-heavy and local data synchronization is part of the job.

## Operational considerations

The project is designed with practical concerns in mind:

- environment-aware configuration
- photo synchronization and local cache usage
- model retrieval from tracked experiment artifacts
- logging of both predictions and metadata
- compatibility with Colab and local machine environments

## Current status

The inference layer is an active part of the project alongside training and tuning. The workflow is designed for local and Colab-based execution; it should not be mistaken for a hosted prediction API.

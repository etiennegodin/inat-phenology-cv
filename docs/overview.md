# Project overview

## The problem

Plant phenology describes recurring life-cycle events such as flowering and fruiting. These events are useful signals for ecological analysis, but observations are visually uneven: one record may include several photos, while the most informative evidence may appear in only one of them. This project explores how to turn those observation-level photo sets into multi-label phenology predictions.

## The approach

Rather than treating every photo as an independent example, the pipeline groups images by observation and predicts at the observation level. It combines configurable vision backbones with pooled or attention-based multiple-instance learning, then supports the full experiment loop around model development:

- structured configuration and runtime validation with Hydra
- training, evaluation, checkpointing, and experiment tracking with MLflow
- parameter search with Optuna and staged backbone unfreezing
- label review support with CleanLab
- validated batch inference with photo synchronization and DuckDB persistence

## Why it matters

The work connects image modeling with the practical data workflow needed to use predictions downstream. Its focus is both methodological, exploring how multiple images contribute to an observation-level prediction, and engineering-oriented, making experiments configurable, traceable, and repeatable.

## Project goals

1. Build and compare multi-label models for multi-image ecological observations.
2. Make backbone and aggregation choices configurable for controlled experiments.
3. Track training runs, metrics, and artifacts so results can be inspected and compared.
4. Carry trained models into a validated batch-scoring workflow.
5. Support iteration through tuning, evaluation, and label-quality review.

## Core terms

- Observation: an ecological record with one or more photos and associated phenology metadata.
- Image bag: the photos attached to one observation and processed together for an observation-level prediction.
- Multi-label classification: one observation may have several active classes, such as flowering and fruiting.
- Progressive unfreezing: a staged fine-tuning strategy that gradually makes backbone blocks trainable.

## Project status

The repository is an active research and engineering project. Its current emphasis is a modular experiment and inference workflow; model quality should be evaluated through the tracked experiments rather than assumed from the system design alone.

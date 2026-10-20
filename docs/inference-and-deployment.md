# Inference and deployment

Inference takes a registered model and turns iNaturalist observations into validated predictions. Today a command takes an observation URL and returns a prediction, and it can run as a batch job locally or on Colab. It isn't a hosted service yet.

The inference code lives in the same package as training but installs separately. A plain `pip install` is enough to run it, and training needs the `[train]` extra.

## Scoring pipeline

1. Fetch observation metadata and photo URLs from iNaturalist.
2. Validate the raw response (shape, observation ID, photo URLs).
3. Sync photos to local storage.
4. Validate the record against the project's rules.
5. Load the registered model from MLflow.
6. Predict on each observation's bag of photos.
7. Apply per-class thresholds and write predictions to DuckDB.

The code is in `src/plant_pheno/inference/` (`sync.py`, `validator.py`, `predictor.py`, `pipeline.py`) and `src/plant_pheno/inat_client/`.

## Two layers of validation

I split validation on purpose:

- **Client-level:** is the API response well-formed, and are the IDs and URLs correct? This happens at acquisition time.
- **Record-level:** do the business rules hold? Pydantic models check this once the photos are synced.

Taxon membership (is this a flowering plant?) is a property on the validated record, not a raising validator. A non-flowering observation isn't corrupt data, it's out of scope, and the pipeline should be able to say so without treating it as an error.

## The registered model

I log the model as an `mlflow.pyfunc.PythonModel` wrapper (`core/pyfunc.py`). My first attempt, `mlflow.pytorch.log_model`, serialized class references with cloudpickle, which broke after refactors. The pyfunc approach is sturdier.

The artifact is:

- a lean checkpoint (model weights plus the parameters needed to rebuild the model),
- a separate JSON file of per-class thresholds, extracted from the evaluation metrics at registration time.

Preprocessing lives inside the artifact, so a caller only provides image paths. The wrapper takes a list of per-observation inputs, which matches how the model handles variable-length bags.

## Prediction store

Predictions go into DuckDB, following the same raw and staged pattern as the rest of the data layer. The serving schema is separate from the training one on purpose, so serving can be deployed without any training data.

Draft tables: observations, photos, models (name, version, thresholds), predictions, and attention weights.

Predictions and attention weights use composite primary keys, `(observation_id, model_id)` and `(observation_id, model_id, class_name)`, instead of a surrogate ID. Re-running a model on an observation overwrites in place via `ON CONFLICT ... DO UPDATE`, so reruns don't pile up duplicates.

## Operational notes

- Config knows about the environment (`local` or `colab`).
- Photo sync uses a local cache.
- The model is pulled from tracked MLflow artifacts, so a prediction can be traced back to a run.

## Status

Inference was finished before the tuning and unfreezing work: the registered model takes an iNaturalist URL through a command and returns a prediction. The database side is still being shaped.

**Next:** hosting it. Wrapping the pipeline in a small API server is the likely route, but I haven't decided yet. I'm also open to keeping it as a command-line tool if a hosted service doesn't earn its complexity.

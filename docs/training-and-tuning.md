# Training and tuning

## Command entry points

The main CLI is `torch_pipe` and is defined in the package entry points.

Common commands:

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status
```

The config layer also supports composition such as:

```bash
torch_pipe -m command=train architecture=mil_pooling,cross_attention
torch_pipe experiment=unfreeze
```

## Training features

The current training stack includes:

- multi-label objective logic
- multi-image batched observations
- class-wise loss aggregation
- per-class performance logging
- early stopping and checkpointing
- stochastic seeding for reproducibility
- mixed precision support where available
- optional progressive unfreezing of backbone stages

## Progressive unfreezing

One of the major recent additions is a progressive unfreeze pathway for the backbone. This is implemented as a staged training mechanism that unlocks deeper blocks over time instead of training the entire backbone at once.

Key ideas in the codebase:

- unfreezing patience and cooldown logic
- stage-based trainable blocks
- warmup learning-rate behavior for newly unlocked stages
- reset and staleness tracking across classes
- checkpoint-aware resume logic

This logic is covered by tests and is a significant change from the earlier baseline roadmap.

## Hyperparameter tuning

The project includes Optuna-based tuning through a dedicated command path. This allows search-space-based tuning over model and optimizer settings using experiment tracking and a reproducible objective.

Typical tuning flows include:

- search-space selection
- sampler and pruner configuration
- review of best parameters
- retraining with the best configuration

## Evaluation and quality controls

The training workflow includes richer monitoring than a simple validation loss. It tracks metrics, observes class-level behavior, and logs attention statistics where relevant. It also supports more advanced quality workflows, including label review and correction steps.

## CleanLab and data quality

The project has moved beyond training-only evaluation and includes label-quality refinement paths. The cleanlab-related logic is not simply a side experiment; it is part of the workflow and is reflected in tests and documentation.

This support is important because ecological labels can be noisy, and observation-level multi-image datasets often benefit from quality-aware review.

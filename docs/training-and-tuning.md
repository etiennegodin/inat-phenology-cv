# Training and tuning

Training is designed as a repeatable experiment loop: compose a run from configuration, train and evaluate the selected model, then compare its tracked outputs with other runs.

## Command entry points

The `torch_pipe` CLI exposes the main training and experiment workflows:

Common commands:

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status
```

Hydra supports multi-run architecture comparisons and named experiment presets:

```bash
torch_pipe -m command=train architecture=mil_pooling,cross_attention
torch_pipe experiment=unfreeze
```

## Training workflow

The workflow supports:

- multi-label objective logic
- multi-image batched observations
- class-wise loss aggregation
- per-class performance logging
- early stopping and checkpointing
- stochastic seeding for reproducibility
- mixed precision support where available
- optional progressive unfreezing of backbone stages

## Progressive unfreezing

Progressive unfreezing offers a staged alternative to either freezing the backbone throughout training or making it fully trainable from the start. As training progresses, selected backbone blocks can be unlocked based on class-level staleness, with cooldown and warmup controls for each stage.

Key ideas in the codebase:

- unfreezing patience and cooldown logic
- stage-based trainable blocks
- warmup learning-rate behavior for newly unlocked stages
- reset and staleness tracking across classes
- checkpoint-aware resume logic

The pathway is integrated with optimizer parameter groups and checkpoint-aware state, and has dedicated regression coverage.

## Hyperparameter tuning

Optuna provides a dedicated tuning path for exploring configured model and optimizer search spaces. The objective and run tracking make trial results available for review before selecting a configuration for a follow-up training run.

Typical iteration:

- search-space selection
- sampler and pruner configuration
- review of best parameters
- retraining with the best configuration

## Evaluation and quality controls

Evaluation tracks class-level behavior alongside aggregate metrics, and logs attention statistics where applicable. CleanLab-based review tools support inspection of potentially problematic labels as a complement to model evaluation.

## CleanLab and data quality

Ecological labels can be noisy, so quality review is a useful part of the experiment loop. CleanLab support helps surface examples for review; it complements domain judgment rather than replacing it.

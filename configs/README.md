# Hydra configs

Composition, not flags: pick an `env`, `command`, and `architecture`, then override any leaf.

```bash
torch_pipe --cfg job
torch_pipe command=train model.backbone=dinov3_vits14 train.epochs=20
torch_pipe command=tune tune.search_space=coarse tune.n_trials=20
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=cleanlab env=colab
torch_pipe command=status
torch_pipe command=options
torch_pipe -m command=train architecture=mil_pooling,cross_attention
torch_pipe experiment=unfreeze
torch_pipe train.progressive_unfreezing.enabled=true train.progressive_unfreezing.patience=5
```

`command=options` lists Hydra groups and registry-backed choices, including
backbones, architectures, Optuna search spaces, samplers, and pruners. Registry
choices are validated before command dispatch, so unsupported values fail early.
Hydra group choices such as `env`, `command`, and `experiment` are also validated
by Hydra during composition.

Progressive-only training controls are grouped under
`train.progressive_unfreezing`. They default to disabled; select
`experiment=unfreeze` or set `train.progressive_unfreezing.enabled=true` to turn
them on. `model.start_unfreezed` separately controls how many backbone blocks
start trainable.

On Colab, `COLAB_GPU` selects `env=colab` unless you pass `env=` yourself.

`hydra.job.chdir` is false so relative data paths stay at the repo CWD. Resolved YAML is written under `outputs/<command>/...`.

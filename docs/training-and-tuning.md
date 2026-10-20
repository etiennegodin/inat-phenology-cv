# Training and tuning

Training is a repeatable experiment loop: compose a run from config, train and evaluate, then compare the tracked outputs against other runs.

## Commands

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status

torch_pipe -m command=train architecture=mil_pooling,cross_attention   # multi-run comparison
torch_pipe experiment=unfreeze                                         # named preset
```

Other commands (`cleanlab`, `predict`, `test`, `list`, `options`) are listed in `configs/command/`. Experiment presets (`baseline`, `smoke_test`, `unfreeze`) live in `configs/experiment/`.

## Training loop

- Multi-label objective with per-class loss and per-class metric logging
- Variable-size observation bags, batched by `max_images` rather than by observation count
- Checkpointing and resume
- Seeding for determinism, and three seeds as my standard for judging variance
- Mixed precision where the hardware allows
- Optional progressive unfreezing

## Metrics and stopping

This is where I made my most useful mistake. My early stopping watched BCE loss, which is sensitive to calibration. For Fruiting, the loss rose while the model was still ranking examples better, so the "divergence" was a metric artifact and not a training problem.

Now:

- Early stopping and the tuning objective use discrimination metrics. The main one is a normalized excess PR-AUC (`pr_norm_excess`), which measures how far PR-AUC sits above what a no-skill classifier would get for that class. This makes classes with different positive rates comparable.
- Each class has its own staleness counter, and training stops only when every class has been stale for `patience` epochs. In other words, training continues as long as the worst-performing class is still improving.
- A `min_delta` guard ignores improvements that are just noise. Its default should come from the observed distribution of epoch-to-epoch deltas, not from a guess.
- The Optuna objective is `pr_norm_excess_macro`, the macro average across classes, computed at the reloaded best checkpoint. The pruner gets the same metric per epoch. This is separate from the per-trial patience logic on purpose.

## Progressive unfreezing

Instead of keeping the backbone frozen or fully trainable from the start, I unlock backbone blocks stage by stage when the head's training stalls. Warmup learning rates apply to newly unlocked stages, and unfreezing state is saved in checkpoints. There is regression test coverage for the pathway (`tests/test_unfreeze_pathway.py`).

**Status:** the mechanism is built. Validating it is on hold until the training-label audit (below) is done, since noisy labels would muddy the comparison.

- The unfreeze trigger rule is an open question. "All three classes stale" is risky because Flower Budding's label noise may stall it forever. A majority rule could fire early because of a saturated class. I plan to settle this with a controlled experiment.
- The stop condition and the unfreeze condition currently share one staleness array. They need independent counters.
- My first unfreezing experiment changed two things at once (`T_max` and `max_images`), which made it uninterpretable against the baseline. The next one will vary only the unfreeze toggle, with equal epochs and equal `max_images`.

## Hyperparameter search

Trials take about an hour on full data, so the budget shapes the design. The next round waits for the label audit, and the plan is a two-stage Optuna search:

1. **Coarse screening** on a ~20% subsample to rule out clearly bad regions.
2. **Refinement** on the full dataset for the survivors.

The coarse search space covers base learning rate (1e-5 to 1e-3, log scale), head dropout, backbone learning-rate decay, attention dropout, attention width, and gated vs. ungated attention.

One thing I found while testing is that the subsample isn't a faithful, cheap proxy. A 20% run takes about half the time of a full run, not a fifth, because it needs more epochs to converge (it hit the 30-epoch cap, which full runs never do). So I treat stage 1 as a filter for bad configurations, not as a source of final values.

Runs are logged as a parent MLflow run per study with a child run per trial.

## Label quality (Cleanlab)

Cleanlab's confident-learning approach flags labels that look wrong, and it needs predictions that were not trained on the example. That leads to a few rules:

- Out-of-fold predictions only. In-sample predictions are invalid because the model memorizes wrong labels.
- Audit the validation set first. My checkpoint predictions on it are already out-of-sample, so it costs nothing.
- For the training set, I'm running a k-fold audit right now, ahead of more tuning and unfreezing, so those experiments start from cleaner labels. It uses a cheaper proxy than the full model: frozen embeddings plus a lightweight classifier per fold.
- Flower Budding deserves caution, since some of the "errors" are real ambiguity. Cleanlab suggests candidates, and I decide.

Corrections are stored as an overlay keyed by observation ID.

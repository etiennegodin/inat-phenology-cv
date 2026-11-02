
## Methodology

Dataset comparison using baseline model parameters

```json
"seed": 42,
"model": {
    "backbone": "bioclip",
    "architecture": "mil_pooling",
    "start_unfreezed": 1,
    "head_neurons": 256,
    "head_outputs": 1,
    "head_dropout_prob": 0.5,
    "attention_neurons": 128,
    "attention_dropout_prob": 0.1,
    "gated": true,
    "proj_dim": 256,
  },
    "train": {
    "epochs": 30,
    "stopping_patience": 3,
    "base_lr": 0.0001,
    "warmup_epochs": 3,
  }
  ```

## Tracking

*All reported metrics are pr_auc_excess*
| Dataset version | Date | macro | Flowering | Fruiting | Flower Budding | MLflow run id |
|---|---|---|---|---|---|---|
|2|09/17/2026|0.776|0.973|0.811|0.545|f87f9c85a35042b1bfaa197ba74cdf8d |
|3|10/23/2026|0.788|0.974|0.843|0.547|d389acfc58b44182a70c72903c3b0e6b |
|4|10/24/2026|0.794 |0.970 |0.830 |0.582|d389acfc58b44182a70c72903c3b0e6b |

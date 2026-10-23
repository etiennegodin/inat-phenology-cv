
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


## Dataset version

### 2. Val partially audited

Visual inspection of flagged labels from val set.

```json
{
  "train": {
    "total_observations": 14223,
    "Flowering_pos_count": 6442,
    "Flowering_pos_ratio": 0.4529,
    "Fruiting_pos_count": 2582,
    "Fruiting_pos_ratio": 0.1815,
    "Flower_Budding_pos_count": 2795,
    "Flower_Budding_pos_ratio": 0.1965,
    "avg_images_per_obs": 2.52
  },
  "val": {
    "total_observations": 1778,
    "Flowering_pos_count": 823,
    "Flowering_pos_ratio": 0.4629,
    "Fruiting_pos_count": 363,
    "Fruiting_pos_ratio": 0.2042,
    "Flower_Budding_pos_count": 363,
    "Flower_Budding_pos_ratio": 0.2042,
    "avg_images_per_obs": 2.53
  }
}
```

---

### 3. 5-fold cross validation

Visual inspection of cross validated dataset.
100 reviews for Flowering and Fruiting


```json
{
  "train": {
    "total_observations": 14186,
    "Flowering_pos_count": 6504,
    "Flowering_pos_ratio": 0.4585,
    "Fruiting_pos_count": 2612,
    "Fruiting_pos_ratio": 0.1841,
    "Flower_Budding_pos_count": 2775,
    "Flower_Budding_pos_ratio": 0.1956,
    "avg_images_per_obs": 2.51
  },
  "val": {
    "total_observations": 1774,
    "Flowering_pos_count": 824,
    "Flowering_pos_ratio": 0.4645,
    "Fruiting_pos_count": 336,
    "Fruiting_pos_ratio": 0.1894,
    "Flower_Budding_pos_count": 352,
    "Flower_Budding_pos_ratio": 0.1984,
    "avg_images_per_obs": 2.53
  }
}
```

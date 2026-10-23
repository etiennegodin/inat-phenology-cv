# inat-phenology-cv

Multi-image, multi-label plant phenology classification from iNaturalist photos. This is a portfolio and learning project: I built it end to end to understand how deep learning systems actually come together, from data and modeling to experiment tracking and inference.

<!-- TODO: add the attention heatmap figure, e.g. docs/img/attention_example.png -->
<!-- ![Attention weights over the photos of one observation](docs/img/attention_example.png) -->

## Why this project exists

I come from VFX, where I spent 8+ years as a lookdev and lighting TD, and I'm moving into computer science with a focus on machine learning and computer vision. I wanted a project where I could learn the full stack of a vision system, not just train one model.

It started small: a frozen ResNet-50 and a binary flowering / not-flowering classifier trained on 736 photos, at roughly 80% validation accuracy. It was meant to enrich a companion project, [inat-obs-scorer](https://github.com/etiennegodin/inat-obs-scorer). Each version since then has been driven by something that broke or didn't make sense in the previous one: multiple photos per observation, attention-based pooling, multi-label targets, better metrics, hardware-aware caching, new backbones and aggregation strategies, a registered model for inference, and now a label audit before the next round of tuning. The [project history](docs/project-history.md) and [lessons learned](docs/lessons-learned.md) cover that path, including the mistakes.

## The problem

Plant phenology is the timing of life-cycle events such as flowering and fruiting. iNaturalist holds a huge number of dated, geolocated plant photos, but the data is awkward for a classifier:

- An **observation** usually has several photos, and the informative one may be a single close-up.
- An observation can carry **several labels at once**, for example flowering and fruiting.
- Labels come from volunteers, so some classes are genuinely ambiguous.

So the unit of prediction here is the observation, not the image. The model reads a variable-length bag of photos and predicts three independent labels: **Flowering**, **Fruiting** and **Flower Budding**.

The dataset covers about 15k observations from Québec, Canada. The current splits hold 12,608 training and 1,576 validation observations.

## How it works

- **Backbone:** a pretrained vision encoder embeds every photo in the bag. I've worked with BioCLIP, BioCLIP 2 and DINOv3, and the backbone is swappable through config.
- **Aggregation:** two strategies, also swappable.
  - **MIL (CLAM-MB):** attention-based multiple-instance learning with a separate attention branch and classifier head per class, all sharing one backbone pass. This started as plain MIL and grew into CLAM-MB. Attention can be gated and uses dropout.
  - **Cross-attention:** a newer alternative for combining the photos in a bag.
- **Outputs:** independent sigmoid outputs per class, with per-class thresholds computed after training.
- **Attention as an explanation:** the attention weights show which photos in an observation drove each prediction.

More detail is in [docs/architecture.md](docs/architecture.md).

## Results

**Status: in progress.** I have numbers from many runs, but they weren't produced under a controlled, comparable protocol, and I'd rather not present them as a benchmark. The comparison I'm running now is a backbone × aggregation grid, with one to three tuned runs per cell and three seeds where I can afford them.

| Backbone | Aggregation | Macro PR-AUC | Flowering | Fruiting | Flower Budding |
|---|---|---|---|---|---|
| BioCLIP | MIL (CLAM-MB) | 0.832 | 0.986 | 0.872 | 0.637 |
| BioCLIP | Cross-attention | – | – | – | – |
| BioCLIP 2 | MIL (CLAM-MB) | – | – | – | – |
| BioCLIP 2 | Cross-attention | – | – | – | – |
| DINOv3 | MIL (CLAM-MB) | 0.771 | 0.974 | 0.799 | 0.540 |
| DINOv3 | Cross-attention | – | – | – | – |

<!-- TODO: trim or reorder the grid to the combinations you actually run, and fill in numbers when the runs are final. -->

What I can say with some confidence so far:

- Metric choice matters a lot. Early stopping on a calibration-sensitive metric (BCE loss) gave misleading signals, and I now stop on discrimination metrics (PR-AUC variants).
- Early comparisons suggest BioCLIP 2 improves on BioCLIP 1 for fruiting and flower budding and makes flower budding much more stable across seeds. The flowering difference isn't distinguishable from noise. I'll treat this as a lead until the controlled grid confirms it.
- Flower Budding is the hardest class and the noisiest. Some of its ambiguity is human-level, so I treat its metrics with caution.


## Installation

The package has two install profiles, because inference needs far less than training:

```bash
git clone https://github.com/etiennegodin/inat-phenology-cv.git
cd inat-phenology-cv

# inference only (the default): load a registered model and score observations
pip install -e .

# full stack: training, tuning, label audits
pip install -e ".[train]"
```

<!-- TODO: add Python version and any required env vars / MLflow server setup. -->

## Usage

The package installs a `torch_pipe` command, composed with [Hydra](https://hydra.cc/). Runs are tracked in MLflow, and the `env` config switches between local and Colab setups. See the [configuration guide](configs/README.md) for how the config groups compose.

**Inference** (default install): give the registered model an iNaturalist observation URL and get a prediction back.

```bash
torch_pipe command=predict

#Example
torch_pipe command=predict \
  predict.model_name=cv_pheno_bioclip \
  predict.model_version=7 \
  predict.urls=["https://www.inaturalist.ca/observations/409673980"]
```

<!-- TODO: replace with the exact predict example, including the URL argument. -->

**Training and experiments** (needs `[train]`):

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status

# compare aggregation strategies in one sweep
torch_pipe -m command=train architecture=mil_pooling,cross_attention

# named experiment preset
torch_pipe experiment=unfreeze
```

Training expects the prepared iNaturalist data, so these commands show how the project is driven rather than reproducing my runs in one step.

## Repository layout

```text
src/plant_pheno/
├── cli.py            # torch_pipe entry point
├── config/           # Hydra composition, registry, search spaces, runtime resolution
├── core/             # backbone integration, models, device handling, MLflow pyfunc wrapper
├── train/            # dataset, batch sampler, workflow, metrics, flow control, persistence
├── tune/             # Optuna objective
├── inference/        # sync, record validation, predictor, scoring pipeline
├── inat_client/      # iNaturalist API clients, fetchers and writers
├── data/             # DuckDB access, SQL, adapters
├── infra/            # logging, seeding, MLflow helpers, networking
├── cleanlab.py       # label-quality audit
└── register.py       # model registration
configs/              # Hydra config groups: env, command, architecture, experiment
tests/                # regression tests for metrics, sampling, unfreezing, validation, determinism
docs/                 # design notes (see below)
```

## Documentation

- [Overview](docs/overview.md): problem framing, goals, scope
- [Dataset](docs/dataset.md): dataset description
- [Architecture](docs/architecture.md): model design and how the system is organized
- [Training and tuning](docs/training-and-tuning.md): metrics, stopping, unfreezing, hyperparameter search, label audits
- [Inference and deployment](docs/inference-and-deployment.md): scoring, validation, persistence
- [Lessons learned](docs/lessons-learned.md): what went wrong and what I changed
- [Project history](docs/project-history.md): how the project got here

## Status and roadmap

Done:

- Training pipeline with CLAM-MB and cross-attention aggregation, and BioCLIP, BioCLIP 2 and DINOv3 backbones
- A registered model that turns an iNaturalist URL into a prediction

In progress:

- A Cleanlab k-fold audit of the training labels, running before any more tuning or unfreezing
- The backbone × aggregation comparison above

Next:

- Progressive backbone unfreezing (built, still to be validated) and a two-stage Optuna search
- Hosting inference, possibly as a small API server (undecided)
- Per-observation prediction agreement across seeds
- Possibly publishing model weights on Hugging Face

## Limitations

- Labels are volunteer-contributed iNaturalist annotations and are noisy.
- The data covers observations from Québec, so I make no claims beyond that region.
- Inference runs as a command today. It isn't a hosted service yet.
- Compute is modest (Colab T4/L4 GPUs), so tuning budgets are small and I design experiments around that.

## Built with

PyTorch, timm, open_clip, Hydra, MLflow, Optuna, Cleanlab, DuckDB, Pydantic.

## Credit and references

- Photos and observations come from [iNaturalist](https://www.inaturalist.org/) open data. Individual photos carry their own licenses.
- BioCLIP: Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life* (CVPR 2024).
- CLAM: Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide images* (2021). This is where the multi-branch attention MIL design comes from.
- Attention-based MIL: Ilse et al., *Attention-based Deep Multiple Instance Learning* (2018).

## License

The code is released under the [MIT License](LICENSE). Datasets and pretrained weights are separate: iNaturalist photos carry their own licenses, and each backbone (BioCLIP, BioCLIP 2, DINOv3) comes with its own terms, so check those before reusing or redistributing anything.

## Author

Étienne, Montreal. [GitHub](https://github.com/etiennegodin)
<!-- TODO: add your full name / contact if you want them on the page -->

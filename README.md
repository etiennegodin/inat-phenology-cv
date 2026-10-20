# inat-phenology-cv

Multi-image, multi-label plant phenology classification from iNaturalist photos. This is a portfolio and learning project: I built it end to end to understand how deep learning systems actually come together, from data and modeling to experiment tracking and serving.

<!-- TODO: add the attention heatmap figure, e.g. docs/img/attention_example.png -->
<!-- ![Attention weights over the photos of one observation](docs/img/attention_example.png) -->

## Why this project exists

I come from VFX, where I spent 8+ years as a lookdev and lighting TD, and I'm moving into computer science with a focus on machine learning and computer vision. I wanted a project where I could learn the full stack of a vision system, not just train one model.

It started small: a frozen ResNet-50 and a binary flowering / not-flowering classifier trained on 736 photos, at roughly 80% validation accuracy. It was meant to enrich a companion project, [inat-obs-scorer](https://github.com/etiennegodin/inat-obs-scorer). Each version since then has been driven by something that broke or didn't make sense in the previous one: multiple photos per observation, attention-based pooling, multi-label targets, better metrics, hardware-aware caching, and now backbone comparisons and progressive unfreezing. The [project history](docs/project-history.md) and [lessons learned](docs/lessons-learned.md) cover that path, including the mistakes.

## The problem

Plant phenology is the timing of life-cycle events such as flowering and fruiting. iNaturalist holds a huge number of dated, geolocated plant photos, but the data is awkward for a classifier:

- An **observation** usually has several photos (about 2.5 on average here), and the informative one may be a single close-up.
- An observation can carry **several labels at once**, for example flowering and fruiting.
- Labels come from volunteers, so some classes are genuinely ambiguous.

So the unit of prediction here is the observation, not the image. The model reads a variable-length bag of photos and predicts three independent labels: **Flowering**, **Fruiting** and **Flower Budding**.

The dataset covers about 15k observations from Québec Canada. The current splits hold 12,608 training and 1,576 validation observations.

## How it works

- **Backbone:** a pretrained vision encoder (BioCLIP ViT-B/16/DINOov3) produces an embedding for every photo in the bag. The backbone is swappable through config.
- **Aggregation:** attention-based multiple-instance learning (CLAM-MB style). Each class has its own attention branch and classifier head, and all branches share one backbone forward pass. Attention can be gated and uses dropout.
- **Outputs:** independent sigmoid outputs per class, with per-class thresholds computed after training.
- **Attention as an explanation:** the attention weights show which photos in an observation drove each prediction.

More detail is in [docs/architecture.md](docs/architecture.md).

## Results

**Status: in progress.** I have numbers from many runs, but they weren't produced under a controlled, comparable protocol, and I'd rather not present them as a benchmark. The comparison I'm running now is a backbone × attention-architecture grid, with one to three tuned runs per cell and three seeds where I can afford them.

| Backbone | Attention architecture | Macro PR-AUC | Flowering | Fruiting | Flower Budding |
|---|---|---|---|---|---|
| BioCLIP | MIL pooling | – | – | – | – |
| BioCLIP | Cross-attention | – | – | – | – |
| BioCLIP 2 | MIL pooling | – | – | – | – |
| BioCLIP 2 | Cross-attention | – | – | – | – |

<!-- TODO: confirm which backbones and architectures belong in the grid (DINOv3? CLAM-MB vs the two configs?) and fill in numbers when the runs are final. -->

What I can say with some confidence so far:

- Metric choice matters a lot. Early stopping on a calibration-sensitive metric (BCE loss) gave misleading signals, and I now stop on discrimination metrics (PR-AUC variants).
- Early comparisons suggest BioCLIP 2 improves on BioCLIP 1 for fruiting and flower budding and makes flower budding much more stable across seeds. The flowering difference isn't distinguishable from noise. I'll treat this as a lead until the controlled grid confirms it.
- Flower Budding is the hardest class and the noisiest. Some of its ambiguity is human-level, so I treat its metrics with caution.

## Quickstart

The package installs a `torch_pipe` command, composed with [Hydra](https://hydra.cc/).

```bash
git clone https://github.com/etiennegodin/inat-phenology-cv.git
cd inat-phenology-cv
pip install -e .
```

```bash
torch_pipe command=train
torch_pipe command=tune tune.search_space=coarse
torch_pipe command=val val.model_name=cv_inat val.model_version=1
torch_pipe command=status

# compare architectures in one sweep
torch_pipe -m command=train architecture=mil_pooling,cross_attention

# named experiment preset
torch_pipe experiment=unfreeze
```

Runs are tracked in MLflow, and the `env` config switches between local and Colab setups. See the [configuration guide](configs/README.md) for how the config groups compose. Training expects the prepared iNaturalist data, so the quickstart is meant to show how the project is driven rather than to reproduce my runs in one step.

<!-- TODO: add Python version and any required env vars / MLflow server setup. -->

## Repository layout

```text
src/plant_pheno/
├── cli.py            # torch_pipe entry point
├── config/           # Hydra composition, registry, search spaces, runtime resolution
├── core/             # backbone integration, MIL model, device handling, MLflow pyfunc wrapper
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
- [Architecture](docs/architecture.md): model design and how the system is organized
- [Training and tuning](docs/training-and-tuning.md): metrics, stopping, unfreezing, hyperparameter search, label audits
- [Inference and deployment](docs/inference-and-deployment.md): batch scoring, validation, persistence
- [Lessons learned](docs/lessons-learned.md): what went wrong and what I changed
- [Project history](docs/project-history.md): how the project got here

## Status and roadmap

Active work:

- Progressive backbone unfreezing, including the open question of when the trigger should fire
- A two-stage Optuna search (coarse screening on a subsample, then refinement on the full dataset)
- The backbone × attention-architecture comparison above
- The serving side: a registered model, a validated scoring pipeline and a DuckDB prediction store

Next up:

- A Cleanlab audit of the training labels using a cheaper k-fold proxy
- Checking prediction agreement across seeds at the per-observation level
- Possibly publishing model weights on Hugging Face

## Limitations

- Labels are volunteer-contributed iNaturalist annotations and are noisy.
- The data covers Northeastern US and Canadian flowering plants, so I make no claims beyond that.
- This is a research and learning codebase, not a hosted prediction service. Inference runs as local or Colab batch jobs.
- Compute is modest (Colab T4/L4 GPUs), so tuning budgets are small and I design experiments around that.

## Built with

PyTorch, timm, open_clip, Hydra, MLflow, Optuna, Cleanlab, DuckDB, Pydantic.

## Credit and references

- Photos and observations come from [iNaturalist](https://www.inaturalist.org/) open data. Individual photos carry their own licenses.
- BioCLIP: Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life* (CVPR 2024).
- CLAM: Lu et al., *Data-efficient and weakly supervised computational pathology on whole-slide images* (2021). This is where the multi-branch attention MIL design comes from.
- Attention-based MIL: Ilse et al., *Attention-based Deep Multiple Instance Learning* (2018).

## License

<!-- TODO: add license -->

## Author

Étienne, Montreal. [GitHub](https://github.com/etiennegodin)
<!-- TODO: add your full name / contact if you want them on the page -->

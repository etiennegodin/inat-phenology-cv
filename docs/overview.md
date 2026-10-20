# Overview

## The problem

Plant phenology is the timing of recurring life-cycle events like flowering and fruiting. It's a useful ecological signal, and iNaturalist is a large source of dated, geolocated plant photos. Turning those photos into phenology labels is harder than it sounds:

- One observation often has several photos, and the useful evidence may be in only one of them.
- An observation can be flowering and fruiting at once, so the labels aren't mutually exclusive.
- Labels are volunteer annotations, and some classes (Flower Budding especially) are ambiguous even to humans.

## The approach

The unit of prediction is the **observation**, not the photo. Photos from one observation form a bag. A pretrained backbone embeds each photo, and an attention-based multiple-instance learning (MIL) head combines them into three independent predictions: Flowering, Fruiting and Flower Budding.

Around the model there's the machinery I wanted to learn properly: configurable experiments, tracked runs, hyperparameter search, staged fine-tuning, label auditing, and a validated batch-scoring path.

## Goals

This is a portfolio and learning project, so the goals are about both the model and the engineering:

1. Build and compare multi-label models for multi-image observations.
2. Keep backbone and aggregation choices configurable so comparisons stay controlled.
3. Track every run so results can be inspected and compared later.
4. Carry a trained model into a validated batch-scoring workflow.
5. Get better at experiment design, which means isolating variables, picking honest metrics, and auditing my own data.

## Scope

- **Data:** about 15k observations from Québec, Canada. Current splits are 12,608 training and 1,576 validation observations.
- **Classes:** Flowering, Fruiting and Flower Budding. Their positive rates differ, which is why I normalize metrics per class (see [training and tuning](training-and-tuning.md)).
- **Out of scope for now:** a hosted prediction service, non-flowering plants, and regions outside the dataset.

## Core terms

- **Observation:** one iNaturalist record with one or more photos and phenology annotations.
- **Bag:** the set of photos from one observation, processed together.
- **Multi-label:** several classes can be active at once.
- **MIL:** multiple-instance learning, where labels exist for a bag but not for each instance in it.
- **Progressive unfreezing:** a staged fine-tuning strategy that gradually makes backbone blocks trainable.

## Where things stand

The project is active. Training, a registered model and command-line inference all work today. Model quality is judged through tracked experiments, not assumed from the design, so a controlled backbone × aggregation comparison is in progress, and the results section of the [README](../README.md) will be filled in once it's done. Before the next round of tuning and unfreezing, I'm auditing the training labels with Cleanlab.

# Lessons learned

A running list of things that went wrong or surprised me, and what I changed. I keep it because the mistakes taught me more than the parts that worked on the first try.

## Judge training by the right metric

BCE loss looked like it was diverging for Fruiting. The model's ranking of examples was still improving. The loss was reacting to calibration, which early stopping shouldn't care about. Switching to a discrimination metric (PR-AUC) fixed the stopping behavior. **Lesson:** make sure the metric measures the thing you want to stop on.

## Compare classes fairly

Raw PR-AUC isn't comparable across classes with different positive rates. A class at 20% positives starts from a very different baseline than one at 45%. Normalizing PR-AUC by what a no-skill classifier would score made macro averages meaningful.

## Isolate variables

My first unfreezing experiment changed the cosine schedule length and `max_images` at the same time, starting from epoch 0. When the result looked different from baseline, I couldn't say why. Experiments now change one thing at a time, and I run three seeds before believing a difference.

## Cleanlab needs out-of-sample predictions

Predictions on training data are useless for finding label errors, because the model has already memorized the mistakes. Validation-set predictions from a saved checkpoint are genuinely out-of-sample, so that's the free first audit. Training-set audits need k-fold.

## A cheap proxy isn't always cheap or faithful

The 20% subsample for hyperparameter screening ran at about half the cost of a full run instead of a fifth, and it needed more epochs to converge. So learning rate is sensitive to the fidelity of the proxy. I use it to eliminate bad configs and not to pick final values.

## Serialize for robustness, not convenience

`mlflow.pytorch.log_model` pickles class references, and it broke after normal refactors. A pyfunc wrapper with a lean checkpoint survives code changes much better.

## Backbones differ in stability, not only accuracy

In early comparisons, BioCLIP 2 improved on BioCLIP for some classes. Another visible difference was much lower variance across seeds for Flower Budding. A mean score alone would have hidden that.

## Still open

- When should unfreezing fire, given that one class's label noise can distort an "all classes stale" rule?
- How much do individual observation predictions agree across seeds? I only look at aggregate metrics now.
- How much of Flower Budding's error is label ambiguity as opposed to model weakness?

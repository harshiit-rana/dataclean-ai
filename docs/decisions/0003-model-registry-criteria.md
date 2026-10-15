# 0003. What a model family must meet to join the registry

- **Status:** Recommendation only, 15 October 2026. The decision is the owner's (issue #52); nothing depends on
  it being accepted except that #53 (Extra Trees) and #56 (stacking) wait for it. If it is accepted as written,
  #53 can proceed.
- **Context:** `modeling.MODELS` holds six `ModelSpec`s, three for each kind of target. Requests to add
  algorithms (XGBoost, LightGBM, CatBoost, SVM, k-NN, Naive Bayes, neural networks) will keep coming. Without
  written criteria each is argued from scratch, and a registry grows for appearance rather than benefit. Each
  family added also makes every comparison longer and, until #35 is settled, raises the AUC bar the verdict
  demands (it rises with the number of candidates), so a family has a price even when it costs nothing to write.

## The bar

A new family joins the registry only if **all** of these hold. They are ordered so that the cheap ones to check
come first.

1. **It is an estimator the pipeline can hold.** A scikit-learn-compatible class that fits on the pipeline's
   inputs (scaled numbers and one-hot categories, blanks already filled), takes string class labels, and gives
   `predict_proba` for a classifier. No special data path.
2. **It reproduces from the exported script.** A `ModelSpec` is `(module, class, params)`, and the script is
   written from those three values. The family must pass `test_every_registered_model_exports_valid_python` and
   the export subprocess test that reproduces the metrics. The dependencies it needs are scikit-learn and pandas
   only, or an explicit dependency decision (see below).
3. **It is reproducible.** A fixed `random_state` gives the same fit, or the family is marked `seeded` so the seed
   check applies to it. A fit that differs between runs for no stated reason cannot be compared fold by fold.
4. **It respects the folds.** It does no internal cross-validation or shuffling of its own that ignores the grouped
   or date-ordered folds the run uses, or if it does (as scikit-learn's `TargetEncoder` does) the interaction is
   written down and tested. A family that quietly leaks across entities or time is worse than no family.
5. **It is affordable.** A fit on 20,000 rows is within a small multiple of the existing families, it has a
   `cost_weight` set from a timing against logistic regression (so the processing limit sees it correctly), and a
   `complexity_rank` (so the one-standard-error rule knows which of two tied models is simpler).
6. **It has a small, curated search space.** A handful of parameters that move a score on tabular data of this size,
   at most a few hundred combinations. A huge grid over a few hundred rows finds the fold's quirk, not a better
   model.
7. **It can be explained, or says it cannot.** Either `feature_importance` works on it, or its spec declares
   `explains_features=False` and the card says so. A misleading ranking is worse than none.
8. **It earns its place on evidence.** Under the existing fold-based comparison on the reference set of tabular files
   (the performance-benchmark issue, once it exists; the project's sample files until then), it wins or ties within
   one standard error often enough to matter, or it brings a different bias that the existing families lack, and it
   does so *after* the verdict's candidate correction is counted. "It is popular" and "other tools have it" are not
   evidence.

## Dependencies

A family that needs a new package is a dependency decision before it is a model decision: licence, install size,
wheels on every platform the app runs on, and whether its results are stable across the library's versions (a
reproducibility promise the export makes on the user's behalf). The default answer is no, and it is revisited with
the benchmark in hand, not before.

## Decisions under these criteria

| Candidate | Decision | Why |
| --- | --- | --- |
| **Extra Trees** (`ExtraTreesClassifier`/`Regressor`) | **Accepted** (#53) | In scikit-learn, so no dependency; same shape as the random forest (criteria 1 to 4, 6, 7 follow from it); usually matches a random forest on tabular data at lower fit cost and lower variance on small noisy files (criteria 5 and 8, to be shown by the one benchmark comparison the issue asks for); placed after the forests in registry order. |
| **XGBoost, LightGBM, CatBoost** | **Deferred** | `HistGradientBoosting*` is the same algorithm family and is already in scikit-learn, so a new dependency would have to beat it on the benchmark first, and none has been shown to. Revisit when the benchmark exists. |
| **SVM** | **Not planned** | Slow to fit as rows grow (fails criterion 5 at 20,000 rows), needs scaling and a kernel choice that the curated search would have to carry, and gives probabilities only through an extra calibration step; a poor fit for the app's one-click comparison. |
| **k-NN** | **Not planned** | Its cost sits in prediction, which the app does for every scored row; it is sensitive to scaling and to many inputs; it stores the training data in the exported model, which is a data-handling cost the others do not carry. |
| **Naive Bayes** | **Not planned** | Its independence assumption makes it a weaker version of what logistic regression already gives, with probabilities that are usually badly calibrated. |
| **Neural networks** | **Not planned** | The tabular evidence for them over boosted trees is weak at this scale, they are hard to reproduce and explain, and they raise the platform question that is deferred. |

## What would change this

- A benchmark set showing a deferred family beating the incumbents by more than the fold spread on real files.
- A decision on #35 that stops the verdict's bar from rising with the candidate count, which lowers the price of each
  family.
- The owner wanting the registry larger for a reason other than accuracy (familiarity, a customer's request). That is
  a legitimate reason, and it should be written here as one rather than argued as a benefit.

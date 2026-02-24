# Model Card — ARIA Landmark Classification (Benchmark Demo)

## What this is
A supervised learning demo that predicts ARIA landmark likelihoods for DOM elements extracted from a website URL, then visualises top candidates as bounding boxes on a screenshot.

## Why Random Forest (vs SVM / single Decision Tree)
Random Forest is a strong default for this feature space (DOM counts + geometry + page aggregates):
- A single Decision Tree is high-variance and tends to overfit; RF reduces variance via bagging.
- RF captures nonlinear interactions without heavy feature engineering.
- RF is less sensitive to feature scaling than SVM and typically needs less tuning.
- RF probability scores are useful for ranking candidates (top-k) for visualisation and clustering.

## Limitations
- Live URL feature extraction approximates the benchmark extraction pipeline; feature drift can affect results.
- Probabilities are confidence scores.
- Live websites don’t have ground-truth labels; evaluation requires benchmark datasets.


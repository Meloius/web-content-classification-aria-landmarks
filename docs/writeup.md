# ARIA Landmarks via DOM Element Classification (Benchmark Demo)

## Motivation
ARIA landmarks (e.g., `navigation`, `main`, `banner`) help assistive technologies support efficient page navigation. In real-world pages, landmarks are often missing or incomplete. This project demonstrates a supervised-learning approach that predicts likely landmark regions by classifying DOM elements using structural and geometric features and visualizing top candidates.

## What this repository demonstrates
This repo is a demo pipeline that takes a live URL and produces:
- a per-element dataset (DOM attributes + bounding box geometry)
- a feature matrix aligned to the benchmark training schema
- (optionally) landmark class probabilities using a trained model
- an overlay visualization highlighting top predictions

> This repository does not ship trained model binaries. The pipeline is designed so model artifacts can be added locally in `./models/`.

## Pipeline
1. Render the page (Playwright): load URL and wait for the DOM to stabilize.
2. Extract elements: for each element, capture tag/role/ARIA attributes, geometry (left/top/width/height), and XPath.
3. Feature construction: convert extracted signals into the benchmark feature schema:
   - per-page tag counts (e.g., `div_count`, `nav_count`, …)
   - page-level bbox statistics (averages/SD)
   - per-element geometry + normalized geometry (`weighted_*`)
4. Prediction (optional): load a trained classifier pipeline to produce per-class probabilities.
5. Visualization: draw top-K predicted candidates per landmark class on the screenshot.

## Why Random Forest (default)
Random Forest is used as the default classifier because it provides strong performance on mixed tabular features and generalizes better than a single Decision Tree (reduced variance via bagging). It models nonlinear interactions between DOM structure and geometry without requiring heavy feature engineering. Compared to SVM, it is less sensitive to feature scaling and typically requires less careful tuning (kernel choice and C/γ), while still producing probability scores useful for ranking candidates for overlay/clustering workflows.

## Evaluation
Benchmark metrics (cross-validation and per-class precision/recall/F1) are summarized in:
- `docs/results.md`

Key points:
- Cross-validation is treated as the primary estimate of generalization.
- Per-class metrics help identify landmark classes that are harder to predict (often due to class imbalance and overlapping structural cues).

## Limitations
- Live URL feature extraction approximates the original benchmark extraction pipeline; feature drift may reduce performance.
- Model outputs are confidence scores, not “accuracy”.
- Live sites lack ground-truth labels; evaluation requires a labeled benchmark.

## How to reproduce
- Run extraction + feature building from a URL.
- To enable predictions, train or obtain model artifacts from the reference benchmark implementation and place them in `./models/`.

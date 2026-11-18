# PathSnap Model 3 Route-Matching System

## Executive summary

This implementation extends the existing Model 3 `HistGradientBoostingClassifier`
with trip-grouped Bayesian hyperparameter optimization, HMM/Viterbi route
reconstruction, exported model artifacts, a local FastAPI service, and a
browser-based evaluator. It does not replace Model 3 with the separate XGBoost
library.

The final model uses the project's 11 engineered candidate features and the
radius-50 labeled data. Bayesian optimization selected a model with a mean
cross-validation observation Top-1 score of **84.89%**. On the held-out
student-test set, this refit achieved **94.62% candidate-row accuracy**,
**89.37% macro F1**, and **84.32% observation Top-1**.

The test Top-1 score is slightly below the original Model 3 artifact's 84.71%;
the optimizer's best cross-validation score is not a guarantee of improved
held-out performance. Results below report this distinction rather than
claiming a test-set gain.

## Data and training

- Training: `outputs/processed/train_labeled_r50.csv` (80 trips).
- Evaluation: `outputs/processed/student_test_labeled_r50.csv` (10 trips).
- Features: 11 label-independent features from `feature_engineering.py`.
- Training cap: 30 complete GPS observations per training trip, retaining all
  candidates for each sampled observation (2,400 observations; 16,737 labeled
  candidate rows). Student-test evaluation uses all 102,522 labeled candidate
  rows.
- Validation: three-fold `GroupKFold`, grouped by trip to avoid trip leakage.
- Search: Optuna TPE sampler, eight trials, maximizing mean observation Top-1.

The compact training cap is intentional for practical local runs; it is recorded
in the model metadata. It should not be mistaken for fitting on all 158,809
training observations.

### Selected parameters

| Parameter | Value |
|---|---:|
| learning rate | 0.04323 |
| maximum iterations | 80 |
| maximum leaf nodes | 61 |
| minimum samples per leaf | 84 |
| L2 regularization | 0.00004174 |

### Measured results

| Metric | Trip-grouped CV mean | Student-test |
|---|---:|---:|
| Observation Top-1 | 84.89% | 84.32% |
| Candidate-row accuracy | 95.01% | 94.62% |
| Macro F1 | 89.27% | 89.37% |
| Log loss | 0.1389 | 0.1442 |

Candidate-row accuracy measures the binary classification of each candidate;
observation Top-1 measures whether the highest-scored candidate is the labeled
road arc. These are different metrics and should not be interchanged.

## Route reconstruction

The inference adapter generates candidates from the selected student's raw GPS
track, applies the same feature engineering used by training, scores candidates
with the saved model, and supplies those emission probabilities to the existing
HMM. The Viterbi decoder selects a temporally coherent candidate sequence;
shortest paths over the directed road graph then fill connecting road arcs.
Long GPS gaps and unreachable transitions are handled as sequence breaks.
Reference route data is not included in API responses or used to make
predictions.

As an offline check, the HMM decoder was run across all 10 student-test trips.
It matched **19,160/19,160** cleaned GPS fixes. Among the **15,866** fixes with a
confident labeled candidate, predicted-arc agreement was **89.72%**. This is a
point-level reachable-candidate score; it is not a route-set overlap metric.

## API and evaluator UI

Run from the repository root after installing `requirements.txt`:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The interface lists student-test trips and
displays the GPS trace, HMM-reconstructed road route, matched points, and
runtime. The UI draws a local SVG and does not send coordinates to third-party
map services.

API endpoints:

- `GET /api/health` — model and decoder status.
- `GET /api/trips` — available student-test trip IDs.
- `POST /api/trips/{trip_id}/match` — reconstruct a route for one trip.

`PATHSNAP_MODEL_PATH` can select an explicit artifact. Otherwise the service
prefers `outputs/models/model_3_bayes_r50.joblib` and falls back to
`outputs/models/model_3.joblib`.

## Reproduction

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m src.bayesian_optimization --radius 50 --folds 3 --trials 8
python -m src.model_3 --radius 50 --folds 3 `
  --params-json outputs\models\model_3_bayes_r50_best_params.json `
  --tag _bayes_r50
python -m unittest discover -s tests -v
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Exported artifacts

The tuned model and its companion artifacts are in `outputs/models/`:

- `model_3_bayes_r50.joblib` — fitted estimator, feature ordering, parameters,
  radius, and metrics.
- `model_3_bayes_r50_best_params.json` — best search parameters.
- `model_3_bayes_r50_trials.csv` — trial-level search history.
- `model_3_bayes_r50_results.json` and `model_3_bayes_r50_cv.csv` — evaluation
  summary and fold-level results.
- `model_3_bayes_r50_confusion_matrix.png` and
  `model_3_bayes_r50_training_curve.png` — diagnostic plots.

## Verification and limitations

The focused integration suite covers API health/UI availability, candidate
probability handoff to the decoder, and Viterbi restart behavior after
impossible transitions. The complete API path was also run successfully for
each student-test trip.

The Windows environment emits a non-fatal joblib warning because `wmic` is not
available to report physical core count; training and inference completed.
Student-test results are a held-out evaluation for this project run, not a
guarantee of performance on new regions or GPS-device conditions.

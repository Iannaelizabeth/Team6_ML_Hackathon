# PathSnap – Distributed Map-Matching and Route Reconstruction Using Machine Learning

Team 6 · Machine Learning Hackathon

PathSnap takes noisy GPS points from a vehicle and works out which roads the vehicle actually drove on. This is called map matching. For each GPS point it finds nearby road segments (candidates) and uses a classifier to score how likely each one is. A Hidden Markov Model (HMM) then picks the most plausible connected route through those candidates.

## How it works

```text
raw trip files → preprocessing → candidate generation (50 m) → feature engineering
→ candidate labelling → classifier training → HMM/Viterbi decoding → maps + API
```

1. **Preprocessing** (`src/preprocessing.py`): reads the `.track`, `.nodes`, `.arcs` and `.route` files, checks coordinates and timestamps, and writes clean tracks. Road geometry is projected into metres before any distance is calculated.
2. **Candidate generation** (`src/candidate_generation.py`): uses a KD-tree (a spatial index for fast nearest-neighbour lookup) to find road segments within 30, 50 or 75 m of each GPS point, keeping at most 10. **50 m is the radius used downstream.**
3. **Feature engineering** (`src/feature_engineering.py`): builds 11 features per (GPS point, candidate) pair:
   - 3 spatial: perpendicular distance, projection ratio, road length
   - 8 motion/context: GPS heading, road heading, heading difference, GPS speed, time difference, distance to previous and next point, connectivity score
4. **Labelling** (`src/labeling.py`): matches each GPS point to the reference route and marks the correct candidate as 1 and the others as 0. Ambiguous points and points that can't be aligned are left out of training.
5. **Classifiers** (models 1–5 below): score each candidate with the probability that it is the right road.
6. **HMM + Viterbi** (`src/hmm.py`, `src/viterbi.py`, `src/route_matching.py`):
   - The hidden states are the candidate segments.
   - The emission score is the classifier's probability.
   - The transition score compares the driving distance along the road graph with the straight-line distance between consecutive GPS points.
   - Viterbi, a dynamic-programming algorithm, finds the highest-scoring sequence of segments.
7. **Visualisation and serving**: Folium maps (`src/maps*.py`) and a FastAPI backend with a browser UI (`src/api.py`, `static/evaluator.html`).

## Models

The task is binary classification: is this candidate the road the vehicle was on? All five models come from scikit-learn.

| # | Model | File | Settings used | Features |
|---|---|---|---|---|
| 1 | Random Forest | `src/models.py`, trained by `src/train_models.py` | 80 trees, max_depth 24, balanced class weights | top 8 |
| 2 | MLP (small neural network) | `src/models.py`, trained by `src/train_models.py` | hidden layers (64, 32), Adam, 20 iterations | top 8 |
| 3 | HistGradientBoosting | `src/model_3.py` | 300 iterations; tuned with Optuna | all 11 |
| 4 | K-Nearest Neighbours | `src/model_4.py` | 25 neighbours, distance weighting | all 11 |
| 5 | Logistic Regression | `src/model_5.py` | L2 regularisation, C = 1.0 | all 11 |

The "top 8" features are chosen by Random Forest importance inside each training fold (`outputs/reports/feature_selection_r50.json`).

**The API and the decoder comparison use the tuned Model 3** (`outputs/models/model_3_tuned.joblib`).

### Evaluation

- **Cross-validation:** 3-fold `GroupKFold` grouped by trip, so points from one trip never appear in both training and validation.
- **Candidate-level metrics:** accuracy, precision, recall, macro F1, log loss.
- **Observation-level Top-1 accuracy:** is the highest-scoring candidate for each GPS point the correct one?
- **Route-level metrics:**
  - route precision, recall and F1, measured by length of matching road
  - route continuity: the share of consecutive matched segments that actually connect
  - candidate miss rate: GPS points whose correct road wasn't among the candidates

### Results (10 student-testing trips)

| Model | CV Macro F1 | Top-1 | Macro F1 | Log loss | Inference |
|---|---|---|---|---|---|
| Random Forest | 0.8761 ± 0.0217 | 0.8479 | 0.8929 | 0.1593 | 0.68 s |
| MLP | 0.8433 ± 0.0209 | 0.8454 | 0.8543 | 0.2102 | 0.21 s |
| HistGradientBoosting | 0.8999 ± 0.0120 | 0.8467 | 0.8965 | 0.1423 | 0.82 s |
| KNN | 0.8693 ± 0.0235 | 0.8208 | 0.8779 | 0.3551 | 21.19 s |
| Logistic Regression | 0.8320 ± 0.0105 | 0.8249 | 0.8371 | 0.2202 | 0.02 s |

| Decoder (tuned Model 3) | Pointwise accuracy | Route F1 | Route continuity |
|---|---|---|---|
| Classifier only | 0.8604 | 0.7461 | 0.9491 |
| HMM, distance-only emission | 0.8739 | 0.9801 | 0.9996 |
| HMM, Model 3 emission | 0.8932 | 0.9837 | 0.9995 |

The full analysis is in the technical report.

## Bayesian optimisation

`src/bayesian_optimization.py` tunes Model 3 with Optuna's TPE sampler (Tree-structured Parzen Estimator), a Bayesian method that picks each new trial based on how earlier trials scored.

- Settings: 30 trials, 3 trip-grouped folds, up to 300 GPS points per trip.
- Objective: cross-validated Top-1 accuracy.
- Parameters searched: `learning_rate`, `max_leaf_nodes`, `min_samples_leaf` and `l2_regularization`.
- `src/compare_tuning.py` compares the tuned and untuned models on identical folds and checks how much the score moves across random seeds.

## Dataset

| File | Contents |
|---|---|
| `.track` | GPS longitude, latitude, timestamp |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Directed road connections (from node → to node) |
| `.route` | Reference sequence of road arc IDs |

| Partition | Trips | Used for |
|---|---:|---|
| `train` | 80 | Training, cross-validation and tuning |
| `student_test` | 10 | Development comparison |
| `evaluator_test` | 10 | Final evaluation only; never used for training or tuning |

The data lives in `Student folder/Student folder/<partition>/<trip_id>/`. **Evaluator-testing data is not in the repo.** It goes only on Laptop 1, in `Student folder/Student folder/evaluator_test/`.

## Project structure

```text
Team6_ML_Hackathon/
├── src/
│   ├── preprocessing.py           data checks, clean tracks, trip discovery
│   ├── candidate_generation.py    30/50/75 m candidate search
│   ├── feature_engineering.py     11 features
│   ├── labeling.py                candidate labels from the reference route
│   ├── data_contract.py           data interface used by model 5 and the HMM
│   ├── evaluation.py              grouped CV, metrics, final RF/MLP fits
│   ├── models.py                  RF and MLP definitions
│   ├── train_models.py            trains/evaluates models 1–2
│   ├── evaluate_student.py        scores RF/MLP on student_test
│   ├── model_runner.py            shared training/evaluation for models 3–4
│   ├── model_3.py                 HistGradientBoosting
│   ├── model_4.py                 KNN
│   ├── model_5.py                 Logistic Regression
│   ├── bayesian_optimization.py   Optuna search for model 3
│   ├── compare_tuning.py          tuned vs untuned model 3
│   ├── hmm.py                     emissions, transitions, road-graph distances
│   ├── viterbi.py                 Viterbi decoder
│   ├── calibrate_hmm.py           fits HMM σ and β on training trips
│   ├── route_matching.py          end-to-end match for one trip
│   ├── route_evaluation.py        route precision/recall/F1/continuity
│   ├── compare_decoders.py        classifier-only vs HMM decoders
│   ├── maps.py                    GPS / road-network / candidate maps
│   ├── maps_advanced.py           candidate maps + static PNGs
│   ├── maps_prediction.py         predicted-route maps
│   └── api.py                     FastAPI backend
├── static/evaluator.html          evaluator UI (Leaflet bundled in static/vendor)
├── tests/                         pytest suite
├── notebooks/eda.ipynb            exploratory data analysis
├── docs/                          data-processing notes, member 2 pipeline, report draft
├── outputs/
│   ├── reports/                   CSV/JSON summaries (EDA, candidates, labels, CV)
│   ├── plots/                     EDA and model-comparison figures
│   ├── models/                    saved models, results JSON, curves, HMM config
│   └── maps/                      Folium HTML maps
├── PathSnap_student_materials/    brief and guidelines
├── requirements.txt
└── README.md
```

These are not committed because they are too large for GitHub:
- `outputs/processed/` (about 700 MB of candidate and feature files)
- `random_forest_r50.joblib` (about 300 MB)
- `mlp_r50.joblib`

Regenerate them with the commands below.

## Setup

Tested with Python 3.13 on Windows.

```powershell
git clone https://github.com/Iannaelizabeth/Team6_ML_Hackathon.git
cd Team6_ML_Hackathon
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest      # only needed for the tests
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

## Reproducing the results

Run from the repository root, in this order. All steps use the 50 m radius by default.

```powershell
# 1. Data checks and candidates
python -m src.preprocessing
python -m src.candidate_generation                       # writes 30, 50 and 75 m candidates

# 2. Features and labels
python -m src.feature_engineering --partition train --radius 50
python -m src.feature_engineering --partition student_test --radius 50
python -m src.labeling --partition train --radius 50
python -m src.labeling --partition student_test --radius 50

# 3. Models 1-2 (Random Forest, MLP)
python -m src.train_models --skip-ablation
python -m src.evaluate_student

# 4. Models 3-5
python -m src.model_3 --max-observations-per-trip 300
python -m src.model_4                                     # KNN, 30 points per trip by default
python -m src.model_5

# 5. Bayesian optimisation of model 3, then the tuned model
python -m src.bayesian_optimization --trials 30 --max-observations-per-trip 300
python -m src.model_3 --max-observations-per-trip 300 --params-json outputs/models/model_3_bayes_r50_best_params.json --tag _tuned
python -m src.compare_tuning

# 6. HMM calibration and decoder comparison
python -m src.calibrate_hmm
python -m src.compare_decoders
```

`src.model_3` processes only 30 points per trip unless you pass `--max-observations-per-trip 300`. The reported Model 3 results use 300.

### Maps

```powershell
python -m src.maps --partition train --trip 00000001 --map-type all
python -m src.maps_advanced --partition train --trip-id 00000001
python -m src.maps_prediction --partition student_test --all
```

Maps are written to `outputs/maps/`. Open the `.html` files in a browser. Each prediction map has layers for the GPS track, the reference route (green), the classifier-only route (orange dashed) and the HMM route (red).

### Tests

```powershell
python -m pytest -q
```

## Deployment (two-laptop evaluation)

| Laptop | Role |
|---|---|
| Laptop 1 | FastAPI backend: tuned Model 3, HMM/Viterbi decoder, road network, server-side metrics |
| Laptop 2 | Evaluator UI: lists the 10 evaluation trips, sends the chosen trip ID to the API, shows the route and metrics |

### Laptop 1 – start the API

```powershell
$env:PATHSNAP_EVAL_PARTITION = "evaluator_test"   # default is student_test
# Optional: $env:PATHSNAP_MODEL_PATH = "outputs/models/model_3_tuned.joblib"
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP with `ipconfig`, and allow port 8000 through Windows Firewall if prompted. Each request is logged in this terminal (`request trip_id=…` and `match trip_id=… pointwise_accuracy=…`). Show these logs to the evaluator.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Model, decoder and partition status |
| `GET /api/trips` | Trip IDs in the served partition |
| `POST /api/trips/{trip_id}/match` | Runs candidates → classifier → HMM/Viterbi and returns the predicted route and metrics. The reference route is never sent back. |

### Laptop 2 – start the UI

```powershell
cd static
python -m http.server 5500
```

Then open `http://localhost:5500/evaluator.html?api=http://<LAPTOP1_IP>:8000`.

CORS (the browser rule that controls which sites may call the API) is open by default. To restrict it, set `PATHSNAP_CORS_ORIGINS` on Laptop 1, for example `http://<LAPTOP2_IP>:5500`.

To check everything on one machine, start the API and open `http://127.0.0.1:8000/`. The backend serves the UI too.

## Team

| SRN | Name |
|---|---|
| PES2UG24CS636 | Ianna Elizabeth Reni |
| PES2UG24CS916 | Abhiram Kolalapudi |
| PES2UG24CS659 | Shloka Shetty |
| PES2UG24CS662 | Swapna |


## Data handling

- Cross-validation is grouped by trip to avoid leakage. Leakage here means points from one trip ending up in both training and validation, which would inflate the scores.
- Scaling and imputation are fitted inside each training fold.
- Evaluator-testing data is never used for training, tuning or feature selection, and is never committed.
- Raw data files are read-only; the code writes everything to `outputs/`.

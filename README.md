# PathSnap – Distributed Map-Matching and Route Reconstruction Using Machine Learning

PathSnap is a machine-learning-based map-matching and route-reconstruction project that identifies the most likely road segments corresponding to noisy GPS observations. It combines geographical data processing, candidate generation, feature engineering, machine-learning classification and sequence-based route reconstruction to produce realistic vehicle trajectories.

## Project Overview

GPS devices provide location observations that may contain errors due to signal interference, urban buildings, multipath effects and irregular sampling intervals. These errors can cause GPS points to appear away from the actual road travelled by a vehicle.

PathSnap addresses this problem by matching GPS observations with possible road segments from a road network and identifying the most likely sequence of road segments.

### Objectives

- Preprocess and validate GPS trajectory and road-network data.
- Generate candidate road segments for each GPS observation.
- Extract geographical and movement-related features.
- Train and compare multiple machine-learning classification models.
- Optimize model hyperparameters using Bayesian Optimization.
- Reconstruct continuous routes using Hidden Markov Models and the Viterbi algorithm.
- Visualize GPS trajectories, road networks and predicted routes using Folium.

## System Workflow

1. **Data Preprocessing:** Clean GPS observations, validate coordinates and timestamps, and check road-network consistency.
2. **Candidate Generation:** Identify nearby road segments for each GPS observation using spatial searching.
3. **Feature Engineering:** Extract features such as perpendicular distance, projection ratio, heading difference, GPS speed and connectivity.
4. **Candidate Labelling:** Assign binary labels to candidate road segments using reference-route alignment.
5. **Model Training:** Train machine-learning models to classify candidate road segments.
6. **Hyperparameter Optimization:** Use Optuna for Bayesian hyperparameter optimization.
7. **Route Reconstruction:** Apply HMM and Viterbi-based sequence decoding to identify a consistent route.
8. **Evaluation and Visualization:** Compare model performance and visualize reconstructed routes.

## Machine Learning Models

The project considers five machine-learning models:

| Model | Algorithm |
|---|---|
| Model 1 | Random Forest |
| Model 2 | MLP Classifier |
| Model 3 | XGBoost |
| Model 4 | RBF Support Vector Machine |
| Model 5 | Logistic Regression |

The models are compared using suitable classification and route-level evaluation metrics.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- Macro F1-score
- Log Loss
- Observation-level Top-1 Accuracy
- Route continuity
- Training and inference time

## Bayesian Optimization

Optuna is used for hyperparameter optimization.

The optimization process explores promising model configurations and compares their performance using a consistent validation strategy.

The objective is to improve model performance while avoiding unnecessary hyperparameter experiments.

## Route Reconstruction

PathSnap includes Hidden Markov Model (HMM) and Viterbi-based route reconstruction.

- **Emission probabilities:** Represent how likely a GPS observation is for a candidate road segment.
- **Transition probabilities:** Represent the likelihood of moving between consecutive candidate road segments.
- **Viterbi algorithm:** Finds the most likely sequence of road segments.

This sequence-based approach helps improve route consistency compared with selecting the best candidate independently for every GPS observation.

## Dataset

The project uses GPS trajectory and road-network information.

The dataset includes:

| File type | Description |
|---|---|
| `.track` | GPS longitude, latitude and timestamp observations |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Connections between road-network nodes |
| `.route` | Reference road-segment sequence |

### Dataset Partitions

| Partition | Number of Trips | Purpose |
|---|---:|---|
| Training | 80 | Model training and cross-validation |
| Student Testing | 10 | Development-stage evaluation |
| Evaluator Testing | 10 | Final evaluation |

Evaluator-testing data must remain separate from model training and hyperparameter optimization.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data processing |
| SciPy | Spatial searching and numerical operations |
| PyProj | Geographical coordinate projection |
| Scikit-learn | Machine-learning algorithms and evaluation |
| XGBoost | Gradient boosting |
| Optuna | Bayesian optimization |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Folium | Interactive geographical maps |
| Jupyter Notebook | Exploratory data analysis and experimentation |
| Git and GitHub | Version control and collaboration |

## Project Structure

```text
Team6_ML_Hackathon/
│
├── docs/
│   └── Project documentation
│
├── notebooks/
│   └── Exploratory data analysis and experiments
│
├── outputs/
│   ├── processed/
│   ├── reports/
│   ├── models/
│   └── figures/
│
├── PathSnap_student_materials/
│   └── Project guidelines and reference materials
│
├── Student folder/
│   └── Student folder/
│       └── GPS trajectory and road-network dataset
│
├── src/
│   ├── preprocessing.py
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── evaluation.py
│   ├── models.py
│   ├── model_3.py
│   ├── model_4.py
│   ├── model_5.py
│   ├── model_runner.py
│   ├── hmm.py
│   ├── viterbi.py
│   ├── maps.py
│   ├── maps_advanced.py
│   └── data_contract.py
│
├── check_route_gaps.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The structure above reflects the main project files and folders. Generated output subfolders may vary depending on the experiments executed.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Iannaelizabeth/Team6_ML_Hackathon.git
```

### 2. Navigate to the Project Directory

```bash
cd Team6_ML_Hackathon
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Install the additional machine-learning dependencies if they are not already included:

```bash
python -m pip install xgboost
```

## Running the Project

Open the repository in Visual Studio Code:

```bash
code .
```

The project contains separate modules for preprocessing, candidate generation, feature engineering, model training, evaluation, route reconstruction and visualization.

Run the appropriate project pipeline or experiment entry point after configuring the dataset paths and required inputs.

### Generating Folium maps

```bash
# GPS + road network + reference route + predicted routes (classifier-only and HMM)
python -m src.maps_prediction --partition student_test --all
# GPS + reference route + candidate segments, plus static trajectory/network PNGs
python -m src.maps_advanced --partition train --trip-id 00000001
```

Maps are written to `outputs/maps/` (open the `.html` files in a browser).

### Running the tests

```bash
python -m pytest -q
```

## Deployment (Two-Laptop Evaluation)

| Laptop | Role |
|---|---|
| Laptop 1 | FastAPI backend: trained model, HMM/Viterbi decoder, road network and server-side evaluation |
| Laptop 2 | Evaluator UI (`static/evaluator.html`): lists the 10 evaluation trips, sends the selected trip ID to the API and shows the returned route and metrics |

### Laptop 1 – start the API

```powershell
# Choose the partition to serve (default: student_test). For the final evaluation:
$env:PATHSNAP_EVAL_PARTITION = "evaluator_test"
# Optional: use a specific model artifact (default: outputs/models/model_3_tuned.joblib)
# $env:PATHSNAP_MODEL_PATH = "outputs/models/model_3_tuned.joblib"

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP address with `ipconfig`, and allow port 8000 through the Windows firewall if prompted. Every inference request is logged in this terminal (`request trip_id=...` and `match trip_id=... pointwise_accuracy=...`); show these logs to the evaluator.

Endpoints:

- `GET  /api/health` – model, decoder and partition status
- `GET  /api/trips` – trip IDs in the served partition
- `POST /api/trips/{trip_id}/match` – runs the full pipeline (candidates → classifier → HMM/Viterbi) and returns the predicted route and server-side metrics. Ground-truth routes are never included in the response.

### Laptop 2 – start the UI

Copy the `static/` folder to Laptop 2 (or clone the repo), then serve it:

```bash
cd static
python -m http.server 5500
```

Open in the browser on Laptop 2:

```
http://localhost:5500/evaluator.html?api=http://<LAPTOP1_IP>:8000
```

The API address is remembered for later visits. CORS is enabled on the API; to restrict it, set `PATHSNAP_CORS_ORIGINS` on Laptop 1 (for example `http://<LAPTOP2_IP>:5500`).

For a single-machine check, open `http://127.0.0.1:8000/` after starting the API. The UI is also served directly by the backend.

## Candidate Generation Experiments

Candidate generation is evaluated using three search radii:

- 30 metres
- 50 metres
- 75 metres

The maximum number of candidates per GPS observation is configured as 10.

The experiments consider candidate coverage, candidate recall, average candidates per observation, zero-candidate observations and processing time.

The final search radius should be selected using measured experimental results.

## Team Contributions

The project is developed collaboratively with responsibilities divided among team members.

| Member | Assigned Responsibility |
|---|---|
| Member 1 | Data preprocessing, candidate generation, feature engineering and labelling |
| Member 2 | Random Forest and MLP Classifier |
| Member 3 | XGBoost, RBF SVM, Bayesian Optimization and model comparison |
| Member 4 | Logistic Regression and route-reconstruction integration |

## Expected Project Outputs

The project is designed to produce:

- Cleaned GPS trajectory data.
- Candidate road-segment datasets.
- Feature-engineered and labelled training data.
- Trained machine-learning models.
- Model evaluation reports.
- Bayesian optimization results.
- Route-reconstruction outputs.
- Interactive geographical maps.
- Model comparison graphs.

Actual performance values and final experimental findings should be reported from the generated evaluation outputs.

## Reproducibility and Data Handling

- Keep raw and processed datasets separate.
- Preserve trip and observation identifiers.
- Use trip-based grouping during cross-validation to avoid data leakage.
- Fit feature scaling within training folds.
- Keep evaluator-testing data separate from training and optimization.
- Record model configurations and experiment results.
- Do not commit unnecessary large datasets, virtual environments or generated model files.

## Repository

GitHub: https://github.com/Iannaelizabeth/Team6_ML_Hackathon

## Conclusion

PathSnap combines machine learning and geographical sequence modelling to address noisy GPS map matching and route reconstruction.

By generating candidate road segments, extracting movement-related features, comparing multiple classification algorithms and applying HMM/Viterbi-based sequence decoding, the project aims to reconstruct realistic vehicle trajectories from GPS observations.

The project provides an integrated framework for studying geographical data processing, machine learning, model optimization and route reconstruction.
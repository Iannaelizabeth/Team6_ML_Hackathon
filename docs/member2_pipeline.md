# Member 2 feature, labeling and model pipeline

## Commands

```powershell
# Needed only if outputs/processed/*_candidates_r*.csv are absent
python -m src.candidate_generation

# Build the training features and labels for each radius
foreach ($r in 30, 50, 75) {
  python -m src.feature_engineering --partition train --radius $r
  python -m src.labeling --partition train --radius $r
}

# Run grouped CV, RF feature ablation/selection, final RF and MLP fits
python -m src.train_models --radius 50 --folds 3

# Optional alternate RF/MLP hyperparameter configurations
python -m src.train_models --radius 50 --folds 3 --sweep

# After training is complete and configurations are fixed, evaluate student-test
python -m src.feature_engineering --partition student_test --radius 50
python -m src.evaluate_student --radius 50
```

Feature engineering streams the trip-grouped candidate files and writes to `outputs/processed/{partition}_features_r{radius}.csv`. If candidate CSVs are absent under `data/candidates/`, the original generator writes them under `outputs/processed/`. Route alignment uses a cached, per-trip route association (`*_route_alignment.csv`) so labeling other radii does not repeat the alignment pass. Training labels are written to `train_labeled_r{radius}.csv`; student-test labels are for a later, development-only evaluation after training and configuration are fixed.

## Feature definitions

Original candidate features (`perpendicular_distance`, `projection_ratio`, `road_length`) are retained. Eight additional predictive features are computed without reading `.route` or labels:

- `gps_heading`: WGS84 bearing from the previous to next GPS fix; at endpoints it uses the one available direction. Duplicate fixes yield missing heading.
- `road_heading`: direction of the candidate's directed arc using its endpoint coordinates in a local azimuthal-equidistant projection.
- `heading_difference`: minimum circular angular difference in `[0, 180]` degrees.
- `gps_speed`: adjacent geodesic displacement divided by positive elapsed time; endpoints use their one available interval.
- `time_difference`: elapsed source time since the preceding fix; the first fix and non-positive intervals are missing.
- `distance_to_previous` / `distance_to_next`: metre distances from the candidate's projected point to the adjacent GPS fix.
- `connectivity_score`: fraction of available adjacent observations for which a directed candidate can join the current candidate at a shared endpoint node. One side is scored only when that adjacent observation has candidates. Score is missing if neither side is available.

Rows are processed per trip, so temporal context and topology never cross trip boundaries. Missing features are imputed inside each model pipeline using training-fold medians. MLP additionally standardizes within the pipeline; the Random Forest is not scaled.

## Route alignment and label semantics

For each GPS observation, the alignment code measures distance to the directed geometries in the chronological `.route`. A forward greedy sequence matcher chooses among nearby route arcs that do not move backward, scoring point-to-segment distance together with disagreement between route-chainage advance and geodesic GPS displacement. It uses the prior selected route state and does not assume observation index equals route index. Invalid route entries are excluded from states; disconnected consecutive route arcs, GPS time gaps over 60 seconds, spatial distance over 75 m, route-order violations and competing route arcs within 5 m are surfaced in status fields.

Only `confident` observations receive ordinary binary labels. A confidently aligned reference arc absent from its candidate list is marked `candidate_miss`; its available candidate rows remain explicit negatives, and the observation has no positive row. Ambiguous, discontinuous, gapped or unalignable observations retain nullable labels and are excluded from training. Candidate coverage is computed only over confident alignments and reported separately from candidate availability.

## Model and validation protocol

The reusable factories are in `src/models.py`. Random Forest uses class-balanced bootstrap weights, configurable tree depth/leaf/features, median imputation and probabilities. MLP uses median imputation, `StandardScaler`, class-balanced sample weights and probabilities. Both expose model configuration and reproducible seeds.

`src/evaluation.py` uses GroupKFold on `trip_id`; candidate rows from a trip cannot cross folds. It reports candidate-row accuracy, precision, recall, macro F1 and log loss, plus observation-level Top-1 accuracy. Top-1 is computed only for observations with a valid positive candidate; candidate misses are excluded from that ranking metric. Held-out permutation importance is reported as a development diagnostic. The model comparison also selects features within each outer training fold: an RF is fit on that fold's training trips, its top eight features are selected, then both classifiers are fitted and scored on the same untouched validation trips. The final retained list is aggregated from training-fold selector rankings. Predefined feature-group ablations use the same GroupKFold procedure and are reported as development comparisons.

The training command writes fold scores, summary metrics, feature missingness/distribution/correlation reports, RF and permutation importance, ablation scores, comparison plots, configurations and final fitted pipelines. Student-test labels are never included in model fitting, feature ablation or cross-validation.

The shared evaluator accepts extra model factories through `cross_validate_models(..., extra_models={name: factory}, models=(...))`; factories receive a random seed and must expose `fit` and `predict_proba`. The roster is RF/MLP (Member 2), XGBoost/RBF SVM (Member 3), and Logistic Regression (Member 4). The other three implementations and their results remain the respective members' work.

## Ownership and limitations

Labeling lives in `src/labeling.py` for this local Member 2 handoff; coordinate this interface with Member 3, who owns shared route-alignment integration. Route alignment is a spatial/temporal heuristic rather than a globally calibrated probabilistic map matcher. Review alignment diagnostics, especially route discontinuities, large GPS gaps and trip 00000054. Model results must be reported only from generated experiment files; synthetic unit-test values are not hackathon metrics.

## Data status

Candidate availability from the existing candidate reports is 90.27%/91.25%/93.47% of training observations and 99.45%/100%/100% of student-test observations at 30/50/75 m. Trip 00000054 is a known outlier from those diagnostics. The candidate inputs were regenerated by the existing generator into ignored `outputs/processed/` because `data/candidates/` was absent in the current workspace.

The current local training label CSVs and diagnostics use the forward monotone matcher. Across confidently alignable training observations, route-candidate coverage is 94.72% (30 m), 96.29% (50 m) and 96.85% (75 m). At 50 m there are 148,609 positive observations, 5,724 explicit candidate misses and 31,180 ambiguous observations; 937,213 candidate rows have binary labels and 290,973 candidate rows have nullable labels. Ambiguous and unalignable diagnostic status counts can overlap.

Trip 00000054 remains an outlier. Route-candidate coverage among its confidently alignable observations is 83.24%, 88.61% and 94.52% at 30/50/75 m, but only 1,844 of 19,313 observations are confidently alignable in the current pass. The existing any-candidate availability rates are 14.05%, 14.47% and 33.24% for this trip. No reference candidate was force-added. No RF/MLP training metrics or saved models have been produced.

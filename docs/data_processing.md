# Data processing, EDA and map outputs

## Reusable entry points

- `src/preprocessing.py` reads `.track`, `.nodes`, `.arcs` and `.route` files and exposes `load_trip`, `preprocess_track`, `validate_network` and `audit_partition`.
- `src/maps.py` exposes `candidate_diagnostics`, `create_trip_map` and `save_static_trip_plots`.
- `notebooks/eda.ipynb` reads the resulting audits and the existing candidate statistics, then produces summary tables and figures.

Run the all-trip data audit with `python -m src.preprocessing`. Run a selected-trip view with `python -m src.maps --partition train --trip-id 00000001 --radius 50`.

## Outputs

- `outputs/reports/{partition}_inventory.csv`: per-trip GPS, network and route checks.
- `outputs/reports/{partition}_consistency.csv`: trip sizes and cross-file ratios.
- `outputs/reports/{partition}_invalid_gps.csv`: invalid GPS rows, retaining original `observation_id` and reason flags.
- `outputs/reports/{partition}_data_quality_summary.json`: partition totals.
- `outputs/reports/candidate_diagnostics_summary.csv` and `candidate_diagnostics_by_trip.csv`: existing 30 m, 50 m and 75 m candidate report analysis.
- `outputs/reports/candidate_data_quality.csv`: candidate schemas and missing-value counts.
- `outputs/processed/`: cleaned GPS track CSVs. Raw files are never rewritten.
- `outputs/maps/`: selected-trip Folium map and static trajectory/network comparison images.
- `outputs/figures/`: EDA figures from the notebook.

## Data findings

The available raw partitions contain 80 training trips and 10 student-test trips (208,417 and 19,160 GPS observations). Across both partitions, no missing values, invalid GPS coordinates, invalid timestamps, duplicate GPS rows, invalid node coordinates, invalid arc endpoints or invalid route arc IDs were found. GPS is chronological for all trips. There are 7,553 non-one-second GPS intervals and 14 broken consecutive reference-route connections (13 training, one student-test). These checks are carried forward rather than silently repairing the data.

Candidate observation coverage is the percentage of observations with at least one candidate, using the existing reports. Training coverage is 90.27%, 91.25% and 93.47% at radii 30 m, 50 m and 75 m; student-test coverage is 99.45%, 100% and 100%. These are candidate-availability rates, not reference-route coverage. A route-to-observation alignment is needed before claiming the correct route arc occurs in a candidate list.

Training trip `00000054` stands out: it has 19,313 observations, while candidate availability is only 14.05%, 14.47% and 33.24% at 30 m, 50 m and 75 m. Its track and network coordinate extents overlap, and its route has no broken connections, so the current checks do not identify a simple missing-network or route-connectivity cause. Keep it visible in label-alignment and map inspection; do not discard it or change the candidate generator based on this diagnostic alone.

All six existing candidate files share the same columns and have no missing candidate values. The report also counts duplicate candidate keys within each CSV chunk; this bounded check does not detect duplicates split across chunk boundaries.

## Feature and labelling integration

Clean tracks preserve the source row index in `observation_id`; candidates use that same identifier. `timestamp` is kept as the source numeric value (the files provide no calendar or timezone metadata). Remove invalid observations only from the clean copy; use the separate invalid-row export to review source-row issues. Keep per-trip boundaries when computing temporal features. Route information belongs in training labels and diagnostics only, never predictive feature columns. Candidate miss/availability flags should be reported separately from negative candidate labels. Trips with route connectivity warnings, large timestamp gaps or unusual candidate miss rates need explicit review during route alignment.

## Map interpretation and limits

The Folium layers show the GPS trajectory, local road-network geometry, reference route and candidate segments. The candidate overlay is capped to the 1,500 closest candidate rows for browser performance. The static network view is clipped around the GPS extent and samples large networks. Neither view repairs the route or establishes ground-truth labels; they are visual QA aids.

Generated inspection examples are `outputs/maps/train_00000001_r50.html` and `outputs/maps/train_00000054_r75.html`, with matching static PNGs for each trip.

import tempfile
import unittest
from pathlib import Path
import joblib

import numpy as np
import pandas as pd

from src.evaluation import grouped_cv_splits
from src.feature_engineering import (angular_difference_degrees,
                                     engineer_trip_candidates,
                                     geodesic_distance_m, track_context)
from src.labeling import align_track_to_route, label_candidate_trip
from src.models import make_mlp, make_random_forest


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame({"longitude": [0.0, .001, .002], "latitude": [0.0, 0.0, 0.0]})
        self.arcs = pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 2]})
        self.track = pd.DataFrame({"longitude": [.0002, .0008, .0012, .0018],
                                   "latitude": [.00001] * 4, "timestamp": [1., 2., 4., 5.],
                                   "observation_id": [0, 1, 2, 3]})

    def test_angular_difference_wraps(self):
        self.assertEqual(float(angular_difference_degrees(359, 1)), 2.0)

    def test_geodesic_distance_in_metres(self):
        distance = float(geodesic_distance_m(0, 0, .001, 0))
        self.assertGreater(distance, 100)
        self.assertLess(distance, 120)

    def test_track_headings_speed_and_irregular_time(self):
        ctx = track_context(self.track)
        self.assertEqual(len(ctx), 4)
        self.assertTrue(ctx.gps_heading.dropna().between(0, 360).all())
        self.assertAlmostEqual(ctx.loc[2, "time_difference"], 2.0)
        self.assertGreater(ctx.loc[1, "gps_speed"], 0)

    def test_duplicate_coordinates_have_missing_heading(self):
        repeated = self.track.iloc[:2].copy()
        repeated.loc[1, ["longitude", "latitude"]] = repeated.loc[0, ["longitude", "latitude"]]
        ctx = track_context(repeated)
        self.assertTrue(np.isnan(ctx.loc[0, "gps_heading"]))
        self.assertTrue(np.isnan(ctx.loc[1, "gps_heading"]))

    def test_all_engineered_columns_and_connectivity(self):
        cand = pd.DataFrame([{"trip_id": "a", "partition": "train", "timestamp": t,
                              "observation_id": i, "candidate_arc_id": aid,
                              "perpendicular_distance": 1., "projection_ratio": .5,
                              "road_length": 100.}
                             for i, (t, aid) in enumerate(zip(self.track.timestamp, [0, 0, 1, 1]))])
        out = engineer_trip_candidates(cand, self.track, self.nodes, self.arcs)
        for col in ["gps_heading", "road_heading", "heading_difference", "gps_speed", "time_difference",
                    "distance_to_previous", "distance_to_next", "connectivity_score"]:
            self.assertIn(col, out)
        self.assertTrue(out.connectivity_score.dropna().between(0, 1).all())
        self.assertEqual(len(out), len(cand))


class LabelingTests(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame({"longitude": [0.0, .001, .002], "latitude": [0.0, 0.0, 0.0]})
        self.arcs = pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 2]})
        self.route = pd.DataFrame({"arc_id": [0, 1]})
        self.track = pd.DataFrame({"longitude": [.0002, .0008, .0012, .0018],
                                   "latitude": [.00001] * 4, "timestamp": [1., 2., 3., 4.],
                                   "observation_id": [0, 1, 2, 3]})

    def test_monotone_route_alignment_and_labels(self):
        aligned = align_track_to_route(self.track, self.nodes, self.arcs, self.route)
        self.assertTrue(aligned.aligned_route_position.is_monotonic_increasing)
        cand = pd.DataFrame([{"trip_id": "x", "partition": "train", "timestamp": t,
                              "observation_id": i, "candidate_arc_id": aid,
                              "perpendicular_distance": 2., "projection_ratio": .5, "road_length": 100.}
                             for i, (t, aid) in enumerate(zip(self.track.timestamp, [0, 0, 1, 1]))])
        labeled, _ = label_candidate_trip(cand, aligned)
        self.assertTrue(set(labeled.label.dropna().astype(int)).issubset({0, 1}))
        self.assertEqual(len(labeled), len(cand))

    def test_candidate_miss_is_explicit_and_ambiguous_not_forced(self):
        alignment = pd.DataFrame({"observation_id": [0, 1], "timestamp": [1., 2.],
                                  "aligned_route_arc_id": [1, 1], "aligned_route_position": [1, 1],
                                  "route_distance_m": [1., 1.], "alignment_status": ["confident", "ambiguous_reference"]})
        candidates = pd.DataFrame({"trip_id": ["x", "x"], "partition": ["train"]*2,
                                   "timestamp": [1., 2.], "observation_id": [0, 1],
                                   "candidate_arc_id": [0, 0], "perpendicular_distance": [2., 2.],
                                   "projection_ratio": [.5, .5], "road_length": [100., 100.]})
        out, diag = label_candidate_trip(candidates, alignment)
        self.assertEqual(out.loc[0, "alignment_status"], "candidate_miss")
        self.assertEqual(int(out.loc[0, "label"]), 0)
        self.assertTrue(pd.isna(out.loc[1, "label"]))
        self.assertFalse(bool(diag.loc[0, "candidate_covered"]))


class ModelPipelineTests(unittest.TestCase):
    def test_group_folds_never_split_trips(self):
        frame = pd.DataFrame({"trip_id": np.repeat([f"t{i}" for i in range(6)], 4),
                              "label": [0, 1] * 12})
        for train, valid in grouped_cv_splits(frame, 3):
            self.assertFalse(set(frame.iloc[train].trip_id) & set(frame.iloc[valid].trip_id))

    def test_model_factories_emit_probabilities(self):
        x = pd.DataFrame({"x": np.arange(40, dtype=float), "z": np.sin(np.arange(40))})
        y = np.array([0, 1] * 20)
        rf = make_random_forest({"n_estimators": 5, "n_jobs": 1}, random_state=1).fit(x, y)
        p = rf.predict_proba(x)
        self.assertEqual(p.shape, (40, 2)); self.assertTrue(np.allclose(p.sum(axis=1), 1))
        mlp = make_mlp({"hidden_layer_sizes": (4,), "max_iter": 3, "batch_size": 8}, random_state=1)
        mlp.fit(x, y, classifier__sample_weight=np.ones(len(y)))
        q = mlp.predict_proba(x)
        self.assertEqual(q.shape, (40, 2)); self.assertTrue(np.allclose(q.sum(axis=1), 1))

    def test_model_pipeline_persistence(self):
        x = pd.DataFrame({"x": np.arange(30, dtype=float)})
        y = np.array([0, 1] * 15)
        model = make_random_forest({"n_estimators": 3, "n_jobs": 1}, random_state=2).fit(x, y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            self.assertTrue(np.allclose(model.predict_proba(x), loaded.predict_proba(x)))


if __name__ == "__main__":
    unittest.main()

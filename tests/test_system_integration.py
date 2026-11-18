import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.data_contract import Trip
from src.hmm import HMMConfig, run_decoder
from src.viterbi import viterbi


class ViterbiIntegrationTests(unittest.TestCase):
    def test_viterbi_restarts_after_impossible_transition(self):
        result = viterbi(
            [np.log([0.8, 0.2]), np.log([0.1, 0.9]), np.log([0.7, 0.3])],
            [np.full((2, 2), -np.inf), np.log([[0.8, 0.2], [0.2, 0.8]])],
        )
        self.assertEqual(result.segments, [(0, 0), (1, 2)])
        self.assertEqual(result.n_breaks, 1)

    def test_hmm_emission_uses_candidate_probabilities(self):
        trip = Trip(
            trip_id="synthetic",
            track=pd.DataFrame({
                "lat": [0.0], "lon": [0.0], "t": [0.0], "x": [0.0], "y": [0.0]
            }),
            nodes_ll=np.array([[0.0, 0.0], [0.0, 0.001]]),
            nodes_xy=np.array([[0.0, 0.0], [111.0, 0.0]]),
            arcs=np.array([[0, 1]]),
            route=None,
            origin=(0.0, 0.0),
            arc_len=np.array([111.0]),
            arc_bearing=np.array([90.0]),
        )
        candidates = pd.DataFrame({
            "point_idx": [0], "arc": [0], "frac_along": [0.0],
            "dist_m": [0.0], "proj_x": [0.0], "proj_y": [0.0],
        })
        result = run_decoder(
            "hmm_classifier", trip, candidates, HMMConfig(),
            {"model": object(), "feature_columns": []}, proba=np.array([0.9])
        )
        self.assertEqual(result.matched["arc"].tolist(), [0])
        self.assertEqual(result.route_arcs, [0])


class ApiIntegrationTests(unittest.TestCase):
    def test_ui_and_health_endpoints(self):
        client = TestClient(app)
        self.assertEqual(client.get("/").status_code, 200)
        health = client.get("/api/health")
        self.assertEqual(health.status_code, 200, health.text)
        self.assertEqual(health.json()["decoder"], "hmm_classifier")
        trips = client.get("/api/trips")
        self.assertEqual(trips.status_code, 200, trips.text)
        self.assertEqual(trips.json()["partition"], "student_test")


if __name__ == "__main__":
    unittest.main()

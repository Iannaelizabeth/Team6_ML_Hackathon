import unittest

import numpy as np

from src.model_runner import route_continuity


class RouteContinuityTests(unittest.TestCase):
    def test_connected_route_has_full_continuity(self):
        arcs = np.array([[0, 1], [1, 2], [2, 3]])
        self.assertEqual(route_continuity([0, 1, 2], arcs), (1.0, 0, 2))

    def test_disconnected_transition_is_counted(self):
        arcs = np.array([[0, 1], [3, 4], [4, 5]])
        self.assertEqual(route_continuity([0, 1, 2], arcs), (0.5, 1, 2))

    def test_single_arc_has_no_discontinuities(self):
        arcs = np.array([[0, 1]])
        self.assertEqual(route_continuity([0], arcs), (1.0, 0, 0))


if __name__ == "__main__":
    unittest.main()

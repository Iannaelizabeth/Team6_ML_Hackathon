"""Server-side route and observation evaluation against held reference routes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.hmm import MatchResult, RoadGraph
from src.labeling import align_track_to_route
from src.preprocessing import load_trip, preprocess_track


def _route_overlap(predicted: list[int], reference: np.ndarray,
                   arc_lengths: np.ndarray) -> tuple[float | None, float | None,
                                                     float | None, float | None]:
    predicted_set = set(map(int, predicted))
    reference_set = set(map(int, reference))
    reference_length = float(sum(arc_lengths[arc] for arc in reference_set))
    true_positive = float(sum(arc_lengths[arc] for arc in predicted_set & reference_set))
    false_positive = float(sum(arc_lengths[arc] for arc in predicted_set - reference_set))
    false_negative = float(sum(arc_lengths[arc] for arc in reference_set - predicted_set))
    predicted_length = true_positive + false_positive
    if reference_length <= 0:
        return None, None, None, None
    precision = true_positive / predicted_length if predicted_length > 0 else 0.0
    recall = true_positive / reference_length
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mismatch = (false_positive + false_negative) / reference_length
    return precision, recall, f1, mismatch


def _route_continuity(matched: pd.DataFrame, graph: RoadGraph) -> tuple[float | None, int]:
    predicted_arcs = matched.sort_values("point_idx", kind="stable").arc.astype(int).tolist()
    if not predicted_arcs:
        return None, 0
    if len(predicted_arcs) == 1:
        return 1.0, 0

    discontinuities = 0
    for arc_a, arc_b in zip(predicted_arcs, predicted_arcs[1:]):
        if arc_a == arc_b:
            continue
        end_node = int(graph.trip.arcs[arc_a, 1])
        start_node = int(graph.trip.arcs[arc_b, 0])
        distance = graph.node_distance(end_node, start_node)
        discontinuities += int(not np.isfinite(distance) or distance > 50.0)
    return float(1.0 - discontinuities / (len(predicted_arcs) - 1)), discontinuities


def evaluate_prediction(
    partition: str,
    trip_id: str,
    result: MatchResult,
    decoder_candidates: pd.DataFrame,
    trip,
) -> dict:
    """Score a completed prediction; reference data is loaded only in this function."""
    if trip.route is not None:
        raise ValueError("Prediction trip must not contain a reference route")

    reference_data = load_trip(partition, trip_id)
    clean_track, _ = preprocess_track(reference_data["track"])
    alignment = align_track_to_route(
        clean_track,
        reference_data["nodes"],
        reference_data["arcs"],
        reference_data["route"],
    )
    confident = alignment.loc[alignment.alignment_status.eq("confident")].copy()
    observation_positions = {
        int(observation_id): position
        for position, observation_id in enumerate(clean_track.observation_id.to_numpy())
    }

    predicted_by_point = (
        result.matched.set_index("point_idx").arc.to_dict()
        if not result.matched.empty else {}
    )
    candidate_arcs_by_point = decoder_candidates.groupby("point_idx").arc.agg(set).to_dict()
    correct = candidate_misses = 0
    for row in confident.itertuples(index=False):
        point_idx = observation_positions.get(int(row.observation_id))
        if point_idx is None:
            continue
        reference_arc = int(row.aligned_route_arc_id)
        correct += int(predicted_by_point.get(point_idx) == reference_arc)
        candidate_misses += int(reference_arc not in candidate_arcs_by_point.get(point_idx, set()))

    evaluable = len(confident)
    pointwise_accuracy = correct / evaluable if evaluable else None
    candidate_miss_rate = candidate_misses / evaluable if evaluable else None

    reference_values = pd.to_numeric(reference_data["route"].arc_id, errors="coerce")
    valid_reference = reference_values.dropna().to_numpy(dtype=float)
    valid_reference = valid_reference[
        (valid_reference % 1 == 0)
        & (valid_reference >= 0)
        & (valid_reference < len(trip.arcs))
    ].astype(int)
    route_precision, route_recall, route_f1, route_mismatch = _route_overlap(
        result.route_arcs, valid_reference, trip.arc_len
    )
    continuity, n_discontinuities = _route_continuity(result.matched, RoadGraph(trip))

    return {
        "pointwise_accuracy": pointwise_accuracy,
        "candidate_miss_rate": candidate_miss_rate,
        "route_precision": route_precision,
        "route_recall": route_recall,
        "route_f1": route_f1,
        "route_mismatch_fraction": route_mismatch,
        "route_continuity": continuity,
        "n_discontinuities": n_discontinuities,
        "evaluable_observations": int(evaluable),
    }

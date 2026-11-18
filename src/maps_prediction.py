"""Folium maps comparing GPS, reference route and the predicted (reconstructed) route.

Layers: original GPS observations, road-network geometry, reference route (where
available), classifier-only prediction and classifier + HMM/Viterbi prediction.
The prediction pipeline itself never sees the reference route
(see ``route_matching.prepare_match``); the reference is only drawn for comparison.

Examples::

    python -m src.maps_prediction --partition student_test --trip-id 00000007
    python -m src.maps_prediction --partition student_test --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import folium
import joblib
from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker
from folium.plugins import Fullscreen

import pandas as pd

from src.preprocessing import PROJECT_ROOT, discover_trips, load_trip
from src.route_matching import _road_network_segments, decode_prepared, prepare_match

MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def _load_artifact(model_path: Path | None = None) -> dict:
    if model_path is None:
        tuned, baseline = MODEL_DIR / "model_3_tuned.joblib", MODEL_DIR / "model_3.joblib"
        model_path = tuned if tuned.is_file() else baseline
    return joblib.load(model_path)


def _route_lines(trip, arc_ids) -> list[list[list[float]]]:
    """Arc ids -> list of [[lat, lon], [lat, lon]] segments."""
    return [trip.arc_latlon(int(a)) for a in arc_ids]


def _add_legend(m: folium.Map, rows: list[tuple[str, str]], title: str) -> None:
    items = "".join(
        f'<div><span style="display:inline-block;width:22px;height:4px;background:{color};'
        f'margin-right:6px;vertical-align:middle"></span>{label}</div>'
        for label, color in rows
    )
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:10px 12px;border:1px solid #999;border-radius:6px;font:13px sans-serif">'
        f'<b>{title}</b>{items}</div>'
    ))


def create_prediction_map(partition: str, trip_id: str, artifact: dict | None = None) -> Path:
    artifact = artifact or _load_artifact()
    prepared = prepare_match(partition, trip_id, artifact)
    hmm = decode_prepared(prepared, "hmm_classifier")
    clf = decode_prepared(prepared, "classifier_only")
    trip = prepared.trip  # reference route removed

    track = trip.track
    points = track[["lat", "lon"]].to_numpy(dtype=float).tolist()
    m = folium.Map(location=[float(track.lat.median()), float(track.lon.median())],
                   zoom_start=15, control_scale=True, tiles="OpenStreetMap")
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(m)

    network = FeatureGroup(name="Road network", show=True)
    segments = [[[lat1, lon1], [lat2, lon2]] for (lon1, lat1), (lon2, lat2) in _road_network_segments(trip)]
    if segments:  # one multi-polyline keeps the HTML small
        PolyLine(segments, color="#8c8c8c", weight=1, opacity=0.45).add_to(network)
    network.add_to(m)

    gps = FeatureGroup(name="GPS observations", show=True)
    PolyLine(points, color="#1f77b4", weight=2, opacity=0.7, tooltip="Raw GPS trajectory").add_to(gps)
    stride = max(1, len(points) // 300)
    for i in range(0, len(points), stride):
        CircleMarker(points[i], radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                     tooltip=f"GPS obs {i}").add_to(gps)
    gps.add_to(m)

    legend = [("Road network", "#8c8c8c"), ("GPS observations", "#1f77b4")]

    # Reference route: loaded separately, only for display/comparison.
    route_df = load_trip(partition, trip_id).get("route")
    reference = ([] if route_df is None else
                 pd.to_numeric(route_df.arc_id, errors="coerce").dropna().astype(int).tolist())
    reference = [a for a in reference if 0 <= a < len(trip.arcs)]
    if reference:
        ref = FeatureGroup(name="Reference route", show=True)
        PolyLine(_route_lines(trip, reference), color="#2ca02c", weight=7, opacity=0.55,
                 tooltip="Reference route").add_to(ref)
        ref.add_to(m)
        legend.append(("Reference route", "#2ca02c"))

    clf_group = FeatureGroup(name="Predicted: classifier only", show=False)
    if clf.route_arcs:
        PolyLine(_route_lines(trip, clf.route_arcs), color="#ff7f0e", weight=3, opacity=0.9,
                 dash_array="6 6", tooltip="Classifier only").add_to(clf_group)
    clf_group.add_to(m)
    legend.append(("Predicted: classifier only (toggle)", "#ff7f0e"))

    hmm_group = FeatureGroup(name="Predicted: classifier + HMM/Viterbi", show=True)
    if hmm.route_arcs:
        PolyLine(_route_lines(trip, hmm.route_arcs), color="#d62728", weight=3.5, opacity=0.95,
                 tooltip="Classifier + HMM/Viterbi").add_to(hmm_group)
    hmm_group.add_to(m)
    legend.append(("Predicted: classifier + HMM/Viterbi", "#d62728"))

    m.fit_bounds([[float(track.lat.min()), float(track.lon.min())],
                  [float(track.lat.max()), float(track.lon.max())]])
    LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    _add_legend(m, legend, f"Trip {trip_id} ({partition})")

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    path = MAP_DIR / f"{partition}_{trip_id}_prediction.html"
    m.save(str(path))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", choices=["train", "student_test"], default="student_test")
    parser.add_argument("--trip-id", default=None)
    parser.add_argument("--all", action="store_true", help="map every trip in the partition")
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()

    artifact = _load_artifact(args.model_path)
    trips = discover_trips(args.partition) if args.all or not args.trip_id else [args.trip_id]
    if not args.all and not args.trip_id:
        trips = trips[:1]
    for trip_id in trips:
        print("Saved:", create_prediction_map(args.partition, trip_id, artifact))


if __name__ == "__main__":
    main()

"""
PathSnap - Interactive Map Visualisation

Creates interactive Folium maps for:
1. GPS trajectories
2. Road networks
3. Candidate road segments

Ground-truth routes are not used for candidate visualisation.
"""

from pathlib import Path
import argparse

import folium
import pandas as pd
from folium.plugins import Fullscreen

from src.preprocessing import (
    PROJECT_ROOT,
    load_trip,
    preprocess_track,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maps"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAP INITIALISATION
# ============================================================

def create_base_map(latitude, longitude, zoom=14):

    map_object = folium.Map(
        location=[float(latitude), float(longitude)],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    # Esri World Street Map - does not require a CartoDB API key.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, HERE, Garmin, USGS, and other contributors",
        name="Esri World Street Map",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(map_object)

    Fullscreen(
        position="topright",
        title="Full Screen",
        title_cancel="Exit Full Screen",
    ).add_to(map_object)

    return map_object


# ============================================================
# GPS TRAJECTORY MAP
# ============================================================

def create_gps_map(partition, trip_id):

    data = load_trip(partition, trip_id)

    track, _ = preprocess_track(data["track"])

    if track.empty:
        raise ValueError(
            "No valid GPS observations found."
        )

    center_lat = track["latitude"].mean()
    center_lon = track["longitude"].mean()

    map_object = create_base_map(
        center_lat,
        center_lon,
    )

    coordinates = track[
        ["latitude", "longitude"]
    ].values.tolist()

    # GPS trajectory.
    folium.PolyLine(
        coordinates,
        color="blue",
        weight=4,
        opacity=0.8,
        tooltip="GPS Trajectory",
    ).add_to(map_object)

    # Starting point.
    folium.Marker(
        coordinates[0],
        popup="Starting Point",
        tooltip="Start",
        icon=folium.Icon(
            color="green",
            icon="play",
        ),
    ).add_to(map_object)

    # Ending point.
    folium.Marker(
        coordinates[-1],
        popup="Ending Point",
        tooltip="End",
        icon=folium.Icon(
            color="red",
            icon="flag",
        ),
    ).add_to(map_object)

    # GPS observation markers.
    # Limit to approximately 300 markers for performance.
    marker_step = max(
        1,
        len(coordinates) // 300,
    )

    observation_layer = folium.FeatureGroup(
        name="GPS Observations",
        show=True,
    )

    for point in coordinates[::marker_step]:

        folium.CircleMarker(
            location=point,
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.6,
            tooltip="GPS Observation",
        ).add_to(observation_layer)

    observation_layer.add_to(map_object)

    folium.LayerControl().add_to(map_object)

    map_object.fit_bounds(coordinates)

    output_path = (
        OUTPUT_DIR / f"{trip_id}_gps.html"
    )

    map_object.save(str(output_path))

    print(f"GPS map saved: {output_path}")
    print(
        f"GPS observations: {len(coordinates)}"
    )

    return output_path


# ============================================================
# ROAD NETWORK MAP
# ============================================================

def create_road_network_map(partition, trip_id):

    data = load_trip(partition, trip_id)

    nodes = data["nodes"]
    arcs = data["arcs"]

    if nodes.empty:
        raise ValueError(
            "No road-network nodes available."
        )

    # Validate node coordinates.
    valid_nodes = nodes.dropna(
        subset=["latitude", "longitude"]
    )

    if valid_nodes.empty:
        raise ValueError(
            "No valid road-network coordinates available."
        )

    # Remove arcs with missing endpoint references.
    valid_arcs = arcs.dropna(
        subset=["from_node_id", "to_node_id"]
    ).copy()

    valid_arcs["from_node_id"] = pd.to_numeric(
        valid_arcs["from_node_id"],
        errors="coerce",
    )

    valid_arcs["to_node_id"] = pd.to_numeric(
        valid_arcs["to_node_id"],
        errors="coerce",
    )

    valid_arcs = valid_arcs.dropna(
        subset=["from_node_id", "to_node_id"]
    )

    valid_arcs["from_node_id"] = (
        valid_arcs["from_node_id"].astype(int)
    )

    valid_arcs["to_node_id"] = (
        valid_arcs["to_node_id"].astype(int)
    )

    # Keep arcs referencing existing nodes.
    valid_arcs = valid_arcs[
        valid_arcs["from_node_id"].between(
            0, len(nodes) - 1
        )
        & valid_arcs["to_node_id"].between(
            0, len(nodes) - 1
        )
    ]

    center_lat = valid_nodes["latitude"].mean()
    center_lon = valid_nodes["longitude"].mean()

    map_object = create_base_map(
        center_lat,
        center_lon,
        zoom=13,
    )

    # Node coordinates: longitude, latitude.
    node_coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy()

    # Represent every road arc as a separate GeoJSON LineString.
    features = []

    for arc_id, arc in valid_arcs.iterrows():

        start_id = int(arc["from_node_id"])
        end_id = int(arc["to_node_id"])

        start = node_coordinates[start_id]
        end = node_coordinates[end_id]

        features.append({
            "type": "Feature",
            "properties": {
                "arc_id": int(arc_id),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [
                        float(start[0]),
                        float(start[1]),
                    ],
                    [
                        float(end[0]),
                        float(end[1]),
                    ],
                ],
            },
        })

    if features:

        road_network_data = {
            "type": "FeatureCollection",
            "features": features,
        }

        folium.GeoJson(
            road_network_data,
            name="Road Network",
            style_function=lambda feature: {
                "color": "gray",
                "weight": 1,
                "opacity": 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["arc_id"],
                aliases=["Arc ID:"],
            ),
        ).add_to(map_object)

    folium.LayerControl().add_to(map_object)

    output_path = (
        OUTPUT_DIR
        / f"{trip_id}_road_network.html"
    )

    map_object.save(str(output_path))

    print(
        f"Road network map saved: {output_path}"
    )

    print(
        f"Valid road segments displayed: {len(features)}"
    )

    return output_path


# ============================================================
# CANDIDATE ROAD SEGMENT MAP
# ============================================================

def create_candidate_map(
    partition,
    trip_id,
    radius=50,
    max_display=1000,
):

    data = load_trip(partition, trip_id)

    track, _ = preprocess_track(data["track"])

    nodes = data["nodes"]
    arcs = data["arcs"]

    if track.empty:
        raise ValueError(
            "No valid GPS observations found."
        )

    if nodes.empty or arcs.empty:
        raise ValueError(
            "Road network nodes or arcs are unavailable."
        )

    candidate_path = (
        PROCESSED_DIR
        / f"{partition}_candidates_r{radius}.csv"
    )

    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    candidates = pd.read_csv(
        candidate_path,
        dtype={"trip_id": str},
    )

    candidates = candidates[
        candidates["trip_id"] == str(trip_id)
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No candidates found for this trip."
        )

    # Limit the number of displayed observations.
    selected_observations = (
        candidates["observation_id"]
        .drop_duplicates()
        .head(max_display)
    )

    candidates = candidates[
        candidates["observation_id"].isin(
            selected_observations
        )
    ]

    candidates["candidate_arc_id"] = pd.to_numeric(
        candidates["candidate_arc_id"],
        errors="coerce",
    )

    candidates = candidates.dropna(
        subset=["candidate_arc_id"]
    )

    candidates["candidate_arc_id"] = (
        candidates["candidate_arc_id"].astype(int)
    )

    center_lat = track["latitude"].mean()
    center_lon = track["longitude"].mean()

    map_object = create_base_map(
        center_lat,
        center_lon,
    )

    # GPS trajectory layer.
    gps_coordinates = track[
        ["latitude", "longitude"]
    ].values.tolist()

    folium.PolyLine(
        gps_coordinates,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="GPS Trajectory",
    ).add_to(map_object)

    # Candidate road segments layer.
    candidate_layer = folium.FeatureGroup(
        name="Candidate Road Segments",
        show=True,
    )

    displayed_arcs = set()

    for arc_id in candidates[
        "candidate_arc_id"
    ].drop_duplicates():

        # Check candidate arc index.
        if arc_id < 0 or arc_id >= len(arcs):
            continue

        arc = arcs.iloc[arc_id]

        if (
            pd.isna(arc["from_node_id"])
            or pd.isna(arc["to_node_id"])
        ):
            continue

        start_id = int(arc["from_node_id"])
        end_id = int(arc["to_node_id"])

        # Validate node references.
        if (
            start_id < 0
            or start_id >= len(nodes)
            or end_id < 0
            or end_id >= len(nodes)
        ):
            continue

        start = nodes.iloc[start_id]
        end = nodes.iloc[end_id]

        if (
            pd.isna(start["latitude"])
            or pd.isna(start["longitude"])
            or pd.isna(end["latitude"])
            or pd.isna(end["longitude"])
        ):
            continue

        segment_coordinates = [
            [
                float(start["latitude"]),
                float(start["longitude"]),
            ],
            [
                float(end["latitude"]),
                float(end["longitude"]),
            ],
        ]

        folium.PolyLine(
            segment_coordinates,
            color="orange",
            weight=2,
            opacity=0.65,
            tooltip=f"Candidate Arc {arc_id}",
        ).add_to(candidate_layer)

        displayed_arcs.add(arc_id)

    candidate_layer.add_to(map_object)

    folium.LayerControl().add_to(map_object)

    map_object.fit_bounds(gps_coordinates)

    output_path = (
        OUTPUT_DIR
        / f"{trip_id}_candidates_r{radius}.html"
    )

    map_object.save(str(output_path))

    print(
        f"Candidate map saved: {output_path}"
    )

    print(
        f"Candidate arcs displayed: {len(displayed_arcs)}"
    )

    return output_path


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="PathSnap interactive map visualisation"
    )

    parser.add_argument(
        "--partition",
        choices=["train", "student_test"],
        default="train",
    )

    parser.add_argument(
        "--trip",
        default="00000001",
    )

    parser.add_argument(
        "--radius",
        type=int,
        choices=[30, 50, 75],
        default=50,
    )

    parser.add_argument(
        "--map-type",
        choices=[
            "gps",
            "network",
            "candidates",
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()

    if args.map_type in ["gps", "all"]:

        create_gps_map(
            args.partition,
            args.trip,
        )

    if args.map_type in ["network", "all"]:

        create_road_network_map(
            args.partition,
            args.trip,
        )

    if args.map_type in ["candidates", "all"]:

        create_candidate_map(
            args.partition,
            args.trip,
            args.radius,
        )


if __name__ == "__main__":
    main()

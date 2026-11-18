
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

from src.preprocessing import DATA_ROOT, load_trip, preprocess_track


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_RADII = [30, 50, 75]
MAX_CANDIDATES = 10


# ============================================================
# LOCAL PROJECTION
# ============================================================

def create_local_transformer(nodes):
    """Create a metre-based local projection centred on the trip."""

    valid = (
        nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
        & nodes[["longitude", "latitude"]].notna().all(axis=1)
    )

    coordinates = nodes.loc[valid, ["longitude", "latitude"]]

    if coordinates.empty:
        raise ValueError("No valid node coordinates available.")

    lon0 = float(coordinates["longitude"].mean())
    lat0 = float(coordinates["latitude"].mean())

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True
    )

    return transformer


def project_coordinates(longitude, latitude, transformer):
    """Convert longitude/latitude into local metre coordinates."""

    return transformer.transform(longitude, latitude)


# ============================================================
# BUILD ROAD SEGMENT INDEX
# ============================================================

def build_segment_index(nodes, arcs, transformer):
    """Build a spatial index over directed road segments."""

    node_x, node_y = project_coordinates(
        nodes["longitude"].to_numpy(dtype=float),
        nodes["latitude"].to_numpy(dtype=float),
        transformer
    )

    node_x = np.asarray(node_x)
    node_y = np.asarray(node_y)

    valid_arcs = (
        arcs["from_node_id"].notna()
        & arcs["to_node_id"].notna()
        & (arcs["from_node_id"] % 1 == 0)
        & (arcs["to_node_id"] % 1 == 0)
        & arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    )

    valid_arc_ids = np.flatnonzero(valid_arcs.to_numpy())

    if len(valid_arc_ids) == 0:
        raise ValueError("No valid road arcs found.")

    from_ids = arcs.iloc[valid_arc_ids][
        "from_node_id"
    ].to_numpy(dtype=int)

    to_ids = arcs.iloc[valid_arc_ids][
        "to_node_id"
    ].to_numpy(dtype=int)

    x1 = node_x[from_ids]
    y1 = node_y[from_ids]
    x2 = node_x[to_ids]
    y2 = node_y[to_ids]

    valid_geometry = (
        np.isfinite(x1)
        & np.isfinite(y1)
        & np.isfinite(x2)
        & np.isfinite(y2)
    )

    valid_arc_ids = valid_arc_ids[valid_geometry]

    x1 = x1[valid_geometry]
    y1 = y1[valid_geometry]
    x2 = x2[valid_geometry]
    y2 = y2[valid_geometry]

    if len(valid_arc_ids) == 0:
        raise ValueError("No valid road segment geometries found.")

    midpoints = np.column_stack((
        (x1 + x2) / 2,
        (y1 + y2) / 2
    ))

    lengths = np.hypot(x2 - x1, y2 - y1)

    tree = cKDTree(midpoints)

    return {
        "tree": tree,
        "arc_ids": valid_arc_ids,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "lengths": lengths,
        "max_half_length": float(lengths.max() / 2)
    }


# ============================================================
# FIND CANDIDATE ROAD SEGMENTS
# ============================================================

def find_candidates(
    gps_x,
    gps_y,
    observation_id,
    segment_index,
    radius,
    max_candidates=MAX_CANDIDATES
):
    """Find the closest directed road segments to one GPS point."""

    search_distance = (
        radius + segment_index["max_half_length"]
    )

    nearby = segment_index["tree"].query_ball_point(
        [gps_x, gps_y],
        search_distance
    )

    if not nearby:
        return []

    indices = np.asarray(nearby, dtype=int)

    x1 = segment_index["x1"][indices]
    y1 = segment_index["y1"][indices]
    x2 = segment_index["x2"][indices]
    y2 = segment_index["y2"][indices]

    dx = x2 - x1
    dy = y2 - y1

    segment_length_squared = dx**2 + dy**2

    denominator = np.where(
        segment_length_squared > 0,
        segment_length_squared,
        1.0
    )

    projection_ratio = (
        (gps_x - x1) * dx
        + (gps_y - y1) * dy
    ) / denominator

    projection_ratio = np.clip(
        projection_ratio,
        0.0,
        1.0
    )

    projected_x = x1 + projection_ratio * dx
    projected_y = y1 + projection_ratio * dy

    perpendicular_distance = np.hypot(
        gps_x - projected_x,
        gps_y - projected_y
    )

    within_radius = perpendicular_distance <= radius

    if not np.any(within_radius):
        return []

    selected = np.flatnonzero(within_radius)

    selected = sorted(
        selected,
        key=lambda j: perpendicular_distance[j]
    )[:max_candidates]

    results = []

    for j in selected:
        segment_position = indices[j]

        results.append({
            "observation_id": int(observation_id),
            "candidate_arc_id": int(
                segment_index["arc_ids"][segment_position]
            ),
            "perpendicular_distance": float(
                perpendicular_distance[j]
            ),
            "projection_ratio": float(
                projection_ratio[j]
            ),
            "road_length": float(
                segment_index["lengths"][segment_position]
            )
        })

    return results


# ============================================================
# GENERATE CANDIDATES FOR ONE TRIP
# ============================================================

def generate_trip_candidates(partition, trip_id, radius):
    """Generate candidates for every valid GPS observation."""

    data = load_trip(partition, trip_id)

    clean_track, _ = preprocess_track(data["track"])

    transformer = create_local_transformer(data["nodes"])

    segment_index = build_segment_index(
        data["nodes"],
        data["arcs"],
        transformer
    )

    gps_x, gps_y = project_coordinates(
        clean_track["longitude"].to_numpy(dtype=float),
        clean_track["latitude"].to_numpy(dtype=float),
        transformer
    )

    gps_x = np.asarray(gps_x)
    gps_y = np.asarray(gps_y)

    valid_gps = np.isfinite(gps_x) & np.isfinite(gps_y)

    rows = []
    candidate_counts = []

    for position, (_, observation) in enumerate(
        clean_track.iterrows()
    ):
        if not valid_gps[position]:
            candidates = []
        else:
            candidates = find_candidates(
                gps_x[position],
                gps_y[position],
                observation["observation_id"],
                segment_index,
                radius
            )

        candidate_counts.append(len(candidates))

        for candidate in candidates:
            rows.append({
                "trip_id": str(trip_id),
                "partition": partition,
                "timestamp": float(observation["timestamp"]),
                **candidate
            })

    result = pd.DataFrame(rows)

    stats = {
        "partition": partition,
        "trip_id": str(trip_id),
        "radius_m": radius,
        "gps_observations": len(clean_track),
        "observations_with_candidates": int(
            sum(count > 0 for count in candidate_counts)
        ),
        "zero_candidate_observations": int(
            sum(count == 0 for count in candidate_counts)
        ),
        "average_candidates_per_observation": (
            float(np.mean(candidate_counts))
            if candidate_counts else 0.0
        ),
        "maximum_candidates_per_observation": (
            max(candidate_counts)
            if candidate_counts else 0
        ),
        "candidate_rows": len(result)
    }

    return result, stats


# ============================================================
# PROCESS A COMPLETE PARTITION
# ============================================================

def process_partition(partition, radius):
    """Generate candidate datasets for all trips in a partition."""

    partition_dir = DATA_ROOT / partition

    trip_dirs = sorted(
        path for path in partition_dir.iterdir()
        if path.is_dir()
    )

    all_candidates = []
    all_stats = []

    print(f"\nPartition: {partition}")
    print(f"Radius: {radius} metres")
    print(f"Trips: {len(trip_dirs)}")

    start_time = time.perf_counter()

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name

        try:
            candidates, stats = generate_trip_candidates(
                partition,
                trip_id,
                radius
            )

            all_candidates.append(candidates)
            all_stats.append(stats)

            print(
                f"{trip_id}: "
                f"{stats['gps_observations']} observations, "
                f"{stats['candidate_rows']} candidates, "
                f"{stats['zero_candidate_observations']} zero-candidate"
            )

        except Exception as error:
            print(f"ERROR processing {trip_id}: {error}")

            all_stats.append({
                "partition": partition,
                "trip_id": trip_id,
                "radius_m": radius,
                "error": str(error)
            })

    elapsed = time.perf_counter() - start_time

    if all_candidates:
        combined = pd.concat(
            all_candidates,
            ignore_index=True
        )
    else:
        combined = pd.DataFrame()

    candidate_path = (
        PROCESSED_DIR
        / f"{partition}_candidates_r{radius}.csv"
    )

    combined.to_csv(candidate_path, index=False)

    stats_df = pd.DataFrame(all_stats)
    stats_df["processing_time_seconds"] = elapsed

    stats_path = (
        REPORT_DIR
        / f"{partition}_candidate_stats_r{radius}.csv"
    )

    stats_df.to_csv(stats_path, index=False)

    print("\nSUMMARY")
    print(f"Candidate rows: {len(combined)}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Candidate dataset: {candidate_path}")
    print(f"Statistics: {stats_path}")

    return combined, stats_df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\nPATHSNAP CANDIDATE GENERATION")
    print("=" * 55)

    for radius in SEARCH_RADII:
        print("\n" + "#" * 55)
        print(f"RUNNING {radius} METRE EXPERIMENT")
        print("#" * 55)

        process_partition("train", radius)
        process_partition("student_test", radius)

    print("\nCANDIDATE GENERATION COMPLETED")

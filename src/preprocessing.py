
from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "Student folder"
    / "Student folder"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FILE READING
# ============================================================

def read_file(path, columns):
    """
    Read a whitespace-separated dataset file.
    Original row indices are preserved.
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns,
        dtype=str
    )


def load_trip(partition, trip_id):
    """
    Load track, nodes, arcs and route files for one trip.
    """

    trip_dir = DATA_ROOT / partition / str(trip_id)

    track = read_file(
        trip_dir / f"{trip_id}.track",
        ["longitude", "latitude", "timestamp"]
    )

    nodes = read_file(
        trip_dir / f"{trip_id}.nodes",
        ["longitude", "latitude"]
    )

    arcs = read_file(
        trip_dir / f"{trip_id}.arcs",
        ["from_node_id", "to_node_id"]
    )

    route = read_file(
        trip_dir / f"{trip_id}.route",
        ["arc_id"]
    )

    # Convert values to numeric.
    # Invalid values become NaN and are reported.
    for column in track.columns:
        track[column] = pd.to_numeric(
            track[column], errors="coerce"
        )

    for column in nodes.columns:
        nodes[column] = pd.to_numeric(
            nodes[column], errors="coerce"
        )

    for column in arcs.columns:
        arcs[column] = pd.to_numeric(
            arcs[column], errors="coerce"
        )

    route["arc_id"] = pd.to_numeric(
        route["arc_id"], errors="coerce"
    )

    return {
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }


# ============================================================
# 3. GPS TRACK PREPROCESSING
# ============================================================

def preprocess_track(track):
    """
    Inspect GPS observations and create a clean copy.

    Original GPS row indices are preserved as observation_id.
    Invalid observations are excluded from the clean copy.
    """

    report = {}

    report["gps_observations"] = len(track)

    report["missing_track_values"] = int(
        track.isna().sum().sum()
    )

    # Check coordinate ranges.
    valid_coordinates = (
        track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
    )

    report["invalid_coordinates"] = int(
        (~valid_coordinates).sum()
    )

    # Check finite timestamps.
    valid_timestamps = np.isfinite(track["timestamp"])

    report["invalid_timestamps"] = int(
        (~valid_timestamps).sum()
    )

    # Keep only valid GPS observations.
    valid_rows = valid_coordinates & valid_timestamps & track.notna().all(axis=1)

    clean_track = track.loc[valid_rows].copy()

    # Preserve the original GPS row number.
    clean_track["observation_id"] = clean_track.index

    invalid = track.loc[~valid_rows].copy()
    invalid["observation_id"] = invalid.index
    invalid["invalid_coordinate"] = ~valid_coordinates.loc[invalid.index]
    invalid["invalid_timestamp"] = ~valid_timestamps.loc[invalid.index]
    invalid["missing_value"] = track.loc[invalid.index].isna().any(axis=1)

    report["valid_observations"] = len(clean_track)
    report["removed_observations"] = (
        len(track) - len(clean_track)
    )

    # Duplicate GPS observations.
    report["duplicate_gps_rows"] = int(
        clean_track.duplicated(
            subset=[
                "longitude",
                "latitude",
                "timestamp"
            ]
        ).sum()
    )

    # Timestamp ordering.
    report["chronological"] = bool(clean_track["timestamp"].is_monotonic_increasing)
    report["duplicate_timestamps"] = int(clean_track["timestamp"].duplicated().sum())
    report["timestamp_span_seconds"] = (
        float(clean_track["timestamp"].max() - clean_track["timestamp"].min())
        if len(clean_track) else 0.0
    )

    # Timestamp intervals.
    time_diff = clean_track["timestamp"].diff()

    report["one_second_intervals"] = int(
        (time_diff == 1).sum()
    )

    report["irregular_intervals"] = int(
        (
            (time_diff != 1)
            & time_diff.notna()
        ).sum()
    )

    report["non_positive_intervals"] = int(
        (time_diff <= 0).sum()
    )

    report["minimum_timestamp_interval"] = (
        float(time_diff.dropna().min())
        if time_diff.notna().any()
        else None
    )

    report["maximum_timestamp_interval"] = (
        float(time_diff.dropna().max())
        if time_diff.notna().any()
        else None
    )

    clean_track.attrs["invalid_observations"] = invalid
    return clean_track, report


# ============================================================
# 4. ROAD NETWORK VALIDATION
# ============================================================

def validate_network(nodes, arcs, route):
    """
    Validate road nodes, directed arcs and reference route.

    Node IDs correspond to zero-based node row indices.
    Arc IDs correspond to zero-based arc row indices.
    """

    report = {}

    report["node_count"] = len(nodes)
    report["arc_count"] = len(arcs)
    report["route_length"] = len(route)
    report["missing_node_values"] = int(nodes.isna().sum().sum())
    report["missing_arc_values"] = int(arcs.isna().sum().sum())
    report["missing_route_values"] = int(route.isna().sum().sum())

    # --------------------------------------------------------
    # Node validation
    # --------------------------------------------------------

    valid_nodes = (
        nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
        & nodes.notna().all(axis=1)
    )

    report["invalid_node_coordinates"] = int(
        (~valid_nodes).sum()
    )

    # --------------------------------------------------------
    # Arc validation
    # --------------------------------------------------------

    valid_arc_values = (
        arcs["from_node_id"].notna()
        & arcs["to_node_id"].notna()
        & (arcs["from_node_id"] % 1 == 0)
        & (arcs["to_node_id"] % 1 == 0)
    )

    valid_arc_endpoints = (
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    )

    valid_arcs = valid_arc_values & valid_arc_endpoints

    report["invalid_arc_endpoints"] = int(
        (~valid_arcs).sum()
    )

    report["self_loop_arcs"] = int(
        (
            valid_arcs
            & (
                arcs["from_node_id"]
                == arcs["to_node_id"]
            )
        ).sum()
    )

    # --------------------------------------------------------
    # Reference route validation
    # --------------------------------------------------------

    valid_route_ids = (
        route["arc_id"].notna()
        & (route["arc_id"] % 1 == 0)
        & route["arc_id"].between(0, len(arcs) - 1)
    )

    report["invalid_route_arc_ids"] = int(
        (~valid_route_ids).sum()
    )

    # --------------------------------------------------------
    # Consecutive route connectivity
    # --------------------------------------------------------

    broken_connections = 0
    checked_connections = 0

    for i in range(len(route) - 1):

        current_id = route.iloc[i]["arc_id"]
        next_id = route.iloc[i + 1]["arc_id"]

        # Do not bridge over invalid route entries.
        if pd.isna(current_id) or pd.isna(next_id):
            continue

        if (
            current_id % 1 != 0
            or next_id % 1 != 0
            or not (0 <= current_id < len(arcs))
            or not (0 <= next_id < len(arcs))
        ):
            continue

        current_arc = arcs.iloc[int(current_id)]
        next_arc = arcs.iloc[int(next_id)]

        if (
            pd.isna(current_arc["from_node_id"])
            or pd.isna(current_arc["to_node_id"])
            or pd.isna(next_arc["from_node_id"])
            or pd.isna(next_arc["to_node_id"])
        ):
            continue

        checked_connections += 1

        if (
            current_arc["to_node_id"]
            != next_arc["from_node_id"]
        ):
            broken_connections += 1

    report["checked_route_connections"] = (
        checked_connections
    )

    report["broken_route_connections"] = (
        broken_connections
    )

    return report


# ============================================================
# 5. PROCESS ONE TRIP
# ============================================================

def process_trip(partition, trip_id):
    """
    Load and validate one trip.

    Returns:
        clean_track
        report dictionary
    """

    data = load_trip(partition, trip_id)

    clean_track, gps_report = preprocess_track(
        data["track"]
    )

    network_report = validate_network(
        data["nodes"],
        data["arcs"],
        data["route"]
    )

    report = {
        "trip_id": str(trip_id),
        "partition": partition,
        **gps_report,
        **network_report
    }

    return clean_track, report


def discover_trips(partition):
    """Return sorted trip directory names, excluding non-directory entries."""
    partition_dir = DATA_ROOT / partition
    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition not found: {partition_dir}")
    return sorted(path.name for path in partition_dir.iterdir() if path.is_dir())


def audit_partition(partition, write_outputs=True):
    """Run raw data-quality checks and emit clean/invalid observation artifacts.

    Raw `.track`, `.nodes`, `.arcs`, and `.route` files are read only. Invalid
    GPS rows are recorded separately with source observation IDs and reason flags.
    """
    inventory = []
    invalid_rows = []
    consistency = []
    for trip_id in discover_trips(partition):
        try:
            data = load_trip(partition, trip_id)
            clean, gps_report = preprocess_track(data["track"])
            network_report = validate_network(data["nodes"], data["arcs"], data["route"])
            row = {"trip_id": trip_id, "partition": partition, **gps_report, **network_report}
            inventory.append(row)
            bad = clean.attrs.get("invalid_observations")
            if bad is not None and not bad.empty:
                bad.insert(0, "partition", partition)
                bad.insert(0, "trip_id", trip_id)
                invalid_rows.append(bad)
            consistency.append({
                "trip_id": trip_id, "partition": partition,
                "track_rows": len(data["track"]), "node_count": len(data["nodes"]),
                "arc_count": len(data["arcs"]), "route_length": len(data["route"]),
                "route_arcs_exceed_network": bool(len(data["route"]) > len(data["arcs"])),
                "nodes_per_arc": float(len(data["nodes"]) / len(data["arcs"])) if len(data["arcs"]) else np.nan,
                "route_arcs_per_gps_point": float(len(data["route"]) / len(data["track"])) if len(data["track"]) else np.nan,
            })
            if write_outputs:
                clean.to_csv(PROCESSED_DIR / f"{partition}_{trip_id}_track.csv", index=False)
        except Exception as exc:
            inventory.append({"trip_id": trip_id, "partition": partition, "error": str(exc)})
    inventory_df = pd.DataFrame(inventory)
    consistency_df = pd.DataFrame(consistency)
    invalid_df = pd.concat(invalid_rows, ignore_index=True) if invalid_rows else pd.DataFrame(
        columns=["trip_id", "partition", "observation_id", "longitude", "latitude", "timestamp", "invalid_coordinate", "invalid_timestamp", "missing_value"]
    )
    if write_outputs:
        REPORT_DIR_LOCAL = PROJECT_ROOT / "outputs" / "reports"
        REPORT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
        inventory_df.to_csv(REPORT_DIR_LOCAL / f"{partition}_inventory.csv", index=False)
        consistency_df.to_csv(REPORT_DIR_LOCAL / f"{partition}_consistency.csv", index=False)
        invalid_df.to_csv(REPORT_DIR_LOCAL / f"{partition}_invalid_gps.csv", index=False)
        _write_quality_summary(inventory_df, consistency_df, partition)
    return inventory_df, consistency_df, invalid_df


def _write_quality_summary(inventory_df, consistency_df, partition):
    """Persist concise aggregate missing-value and data-quality findings."""
    successful = inventory_df.loc[inventory_df.get("error", pd.Series(index=inventory_df.index, dtype=object)).isna()]
    metrics = ["gps_observations", "valid_observations", "missing_track_values", "missing_node_values",
               "missing_arc_values", "missing_route_values", "invalid_coordinates",
               "invalid_timestamps", "duplicate_gps_rows", "irregular_intervals", "invalid_node_coordinates",
               "invalid_arc_endpoints", "invalid_route_arc_ids", "broken_route_connections"]
    summary = {"partition": partition, "trip_count": len(inventory_df), "successful_trips": len(successful)}
    for metric in metrics:
        if metric in successful:
            vals = pd.to_numeric(successful[metric], errors="coerce")
            summary[f"{metric}_total"] = int(vals.sum())
            summary[f"{metric}_mean_per_trip"] = float(vals.mean()) if vals.notna().any() else np.nan
    summary["trips_with_broken_route_connections"] = int((pd.to_numeric(successful.get("broken_route_connections", 0), errors="coerce") > 0).sum())
    summary["trips_with_non_chronological_gps"] = int((~successful.get("chronological", pd.Series(dtype=bool)).fillna(False)).sum())
    summary["median_nodes_per_arc"] = float(consistency_df["nodes_per_arc"].median()) if not consistency_df.empty else np.nan
    path = PROJECT_ROOT / "outputs" / "reports" / f"{partition}_data_quality_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ============================================================
# 6. PROCESS ALL TRIPS IN A PARTITION
# ============================================================

def process_partition(partition):
    """
    Process every trip in a partition.

    Each trip is handled independently so that one
    problematic trip does not stop the entire process.
    """

    partition_dir = DATA_ROOT / partition

    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Partition not found: {partition_dir}"
        )

    trip_dirs = sorted(
        path
        for path in partition_dir.iterdir()
        if path.is_dir()
    )

    reports = []
    failed_trips = []

    print(f"\nProcessing partition: {partition}")
    print(f"Number of trips: {len(trip_dirs)}\n")

    for trip_dir in trip_dirs:

        trip_id = trip_dir.name

        print(f"Processing trip {trip_id}...")

        try:
            clean_track, report = process_trip(
                partition,
                trip_id
            )

            reports.append(report)

            # Save cleaned GPS observations separately.
            clean_track.to_csv(
                PROCESSED_DIR
                / f"{partition}_{trip_id}_track.csv",
                index=False
            )

            print(
                f"  Success: "
                f"{report['valid_observations']} valid GPS points"
            )

        except Exception as error:

            print(f"  ERROR: {error}")

            failed_trips.append({
                "trip_id": trip_id,
                "partition": partition,
                "error": str(error)
            })

            reports.append({
                "trip_id": trip_id,
                "partition": partition,
                "error": str(error)
            })

    report_df = pd.DataFrame(reports)

    # Save partition-level inventory.
    report_df.to_csv(
        OUTPUT_DIR / f"{partition}_inventory.csv",
        index=False
    )

    # Save errors separately.
    if failed_trips:
        pd.DataFrame(failed_trips).to_csv(
            OUTPUT_DIR / f"{partition}_failed_trips.csv",
            index=False
        )

    print(f"\nCompleted partition: {partition}")
    print(f"Successful trips: {len(trip_dirs) - len(failed_trips)}")
    print(f"Failed trips: {len(failed_trips)}")

    return report_df


# ============================================================
# 7. SUMMARY REPORT
# ============================================================

def print_summary(report_df, partition):
    """
    Display a summary of the processed partition.
    """

    successful = report_df[
        report_df["error"].isna()
    ].copy() if "error" in report_df.columns else report_df.copy()

    print("\n" + "=" * 55)
    print(f"{partition.upper()} DATASET SUMMARY")
    print("=" * 55)

    print(f"Total trips: {len(report_df)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed trips: {len(report_df) - len(successful)}")

    if successful.empty:
        print("No successful trips available for summary.")
        return

    numeric_columns = [
        "gps_observations",
        "valid_observations",
        "missing_track_values",
        "invalid_coordinates",
        "invalid_timestamps",
        "irregular_intervals",
        "node_count",
        "arc_count",
        "route_length",
        "invalid_arc_endpoints",
        "invalid_route_arc_ids",
        "broken_route_connections"
    ]

    for column in numeric_columns:

        if column in successful.columns:
            values = pd.to_numeric(
                successful[column],
                errors="coerce"
            )

            print(
                f"{column}: "
                f"total={values.sum():.0f}, "
                f"mean={values.mean():.2f}"
            )


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\nPATHSNAP DATA PREPROCESSING")
    print("=" * 55)

    # Full audit emits clean tracks, invalid-row reason records, inventories,
    # consistency tables and aggregate quality summaries per partition.
    train_report, _, _ = audit_partition("train")
    print_summary(train_report, "train")
    test_report, _, _ = audit_partition("student_test")
    print_summary(test_report, "student_test")

    print("\n" + "=" * 55)
    print("PREPROCESSING COMPLETED")
    print("=" * 55)

    print("\nGenerated files:")
    print("1. outputs/reports/train_inventory.csv")
    print("2. outputs/reports/student_test_inventory.csv")
    print("3. outputs/reports/train_failed_trips.csv (if errors occur)")
    print("4. outputs/reports/student_test_failed_trips.csv (if errors occur)")
    print("5. outputs/processed/ - cleaned GPS track CSV files")

    print("\nOriginal dataset files were not modified.")

"""Trip trajectory and road-network visualisations for PathSnap.

Examples::

    python -m src.maps --partition train --trip-id 00000001 --radius 50

Maps are written under ``outputs/maps``. Raw source files remain read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import PROJECT_ROOT, load_trip, preprocess_track

MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
# candidate_generation.py writes {partition}_candidates_r{radius}.csv here
CANDIDATE_DIR = PROJECT_ROOT / "outputs" / "processed"


def _arc_coordinates(nodes: pd.DataFrame, arcs: pd.DataFrame, arc_ids):
    """Yield valid directed arc coordinate pairs, retaining route order."""
    for arc_id in arc_ids:
        if pd.isna(arc_id) or int(arc_id) != arc_id or not 0 <= int(arc_id) < len(arcs):
            continue
        arc = arcs.iloc[int(arc_id)]
        if pd.isna(arc.from_node_id) or pd.isna(arc.to_node_id):
            continue
        a, b = int(arc.from_node_id), int(arc.to_node_id)
        if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
            continue
        p, q = nodes.iloc[a], nodes.iloc[b]
        if np.isfinite([p.longitude, p.latitude, q.longitude, q.latitude]).all():
            yield [float(p.latitude), float(p.longitude)], [float(q.latitude), float(q.longitude)]


def _network_lines_in_bounds(nodes: pd.DataFrame, arcs: pd.DataFrame, bounds, limit=6000):
    """Vector-filter network geometries to a local map extent before rendering."""
    xy = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    ends = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    valid = np.isfinite(ends).all(axis=1) & (ends % 1 == 0).all(axis=1)
    valid &= (ends[:, 0] >= 0) & (ends[:, 0] < len(nodes)) & (ends[:, 1] >= 0) & (ends[:, 1] < len(nodes))
    ids = np.flatnonzero(valid)
    a, b = ends[ids].astype(int).T
    p, q = xy[a], xy[b]
    valid_xy = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    valid_xy &= (p[:, 0] >= bounds[0]) & (p[:, 0] <= bounds[1]) & (q[:, 0] >= bounds[0]) & (q[:, 0] <= bounds[1])
    valid_xy &= (p[:, 1] >= bounds[2]) & (p[:, 1] <= bounds[3]) & (q[:, 1] >= bounds[2]) & (q[:, 1] <= bounds[3])
    p, q = p[valid_xy], q[valid_xy]
    if len(p) > limit:
        take = np.linspace(0, len(p)-1, limit, dtype=int)
        p, q = p[take], q[take]
    return p, q


def create_trip_map(partition: str, trip_id: str, radius: int = 50, candidate_limit: int = 1500) -> Path:
    """Create a Folium map with GPS, reference route, network and candidates.

    Candidate lines are capped for browser responsiveness; the complete
    candidate dataset is preserved and the cap is reported in the HTML legend.
    """
    try:
        import folium
        from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker
    except ImportError as exc:
        raise RuntimeError("Folium is required for interactive maps (pip install folium).") from exc

    trip_id = str(trip_id)
    data = load_trip(partition, trip_id)
    clean, _ = preprocess_track(data["track"])
    points = clean[["latitude", "longitude"]].to_numpy(dtype=float)
    if len(points) == 0:
        raise ValueError(f"Trip {trip_id} has no valid GPS observations")
    center = [float(clean.latitude.median()), float(clean.longitude.median())]
    lat_pad = max(0.002, float(clean.latitude.max() - clean.latitude.min()) * 0.08)
    lon_pad = max(0.002, float(clean.longitude.max() - clean.longitude.min()) * 0.08)
    bounds = (float(clean.longitude.min()) - lon_pad, float(clean.longitude.max()) + lon_pad,
              float(clean.latitude.min()) - lat_pad, float(clean.latitude.max()) + lat_pad)
    # Esri street tiles: OpenStreetMap blocks tiles on pages opened from disk (file://),
    # and CARTO basemaps now need an API key.
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles=None)
    from src.maps import add_basemaps
    add_basemaps(m)

    gps_group = FeatureGroup(name="GPS observations", show=True)
    # Use a line for the trajectory plus sparse points for reasonable file size.
    PolyLine(points.tolist(), color="#1f77b4", weight=3, opacity=0.85, tooltip="GPS trajectory").add_to(gps_group)
    stride = max(1, len(clean) // 250)
    for _, obs in clean.iloc[::stride].iterrows():
        CircleMarker([obs.latitude, obs.longitude], radius=2, color="#135a9c", fill=True,
                     tooltip=f"observation {int(obs.observation_id)} | t={obs.timestamp:g}").add_to(gps_group)
    gps_group.add_to(m)

    network_group = FeatureGroup(name="Road network", show=False)
    net_start, net_end = _network_lines_in_bounds(data["nodes"], data["arcs"], bounds)
    for p, q in zip(net_start, net_end):
        PolyLine([[p[1], p[0]], [q[1], q[0]]], color="#888888", weight=1, opacity=0.25).add_to(network_group)
    network_group.add_to(m)

    route_group = FeatureGroup(name="Reference route", show=True)
    route_ids = data["route"].arc_id.dropna().tolist()
    for start, end in _arc_coordinates(data["nodes"], data["arcs"], route_ids):
        PolyLine([start, end], color="#d62728", weight=4, opacity=0.85).add_to(route_group)
    route_group.add_to(m)

    cand_path = CANDIDATE_DIR / f"{partition}_candidates_r{radius}.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {cand_path}")
    # Filter a single trip in chunks so multi-trip candidate files remain bounded in memory.
    chunks = []
    for chunk in pd.read_csv(cand_path, dtype={"trip_id": str}, chunksize=250_000):
        selected = chunk.loc[chunk.trip_id == trip_id]
        if not selected.empty:
            chunks.append(selected)
    candidates = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    cand_group = FeatureGroup(name=f"Candidate segments (r={radius}m)", show=False)
    shown = candidates.nsmallest(candidate_limit, "perpendicular_distance") if len(candidates) > candidate_limit else candidates
    for arc_id in shown.candidate_arc_id:
        for start, end in _arc_coordinates(data["nodes"], data["arcs"], [arc_id]):
            PolyLine([start, end], color="#2ca02c", weight=2, opacity=0.45).add_to(cand_group)
    cand_group.add_to(m)
    LayerControl(collapsed=False).add_to(m)
    folium.LatLngPopup().add_to(m)

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    path = MAP_DIR / f"{partition}_{trip_id}_r{radius}.html"
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;'
        f'border:1px solid #aaa">GPS blue -+ reference red -+ candidates green -+ network gray<br>'
        f'Candidates shown: {len(shown):,} / {len(candidates):,}</div>'
    ))
    m.save(str(path))
    return path


def save_static_trip_plots(partition: str, trip_id: str, radius: int = 50) -> list[Path]:
    """Save compact PNG trajectory and road-network comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    trip_id = str(trip_id)
    d = load_trip(partition, trip_id)
    track, _ = preprocess_track(d["track"])
    nodes, arcs = d["nodes"], d["arcs"]
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(track.longitude, track.latitude, color="#1f77b4", linewidth=1.1, label="GPS trajectory")
    for s, e in _arc_coordinates(nodes, arcs, d["route"].arc_id.tolist()):
        ax.plot([s[1], e[1]], [s[0], e[0]], color="#d62728", linewidth=1.5, alpha=.8)
    ax.scatter(track.longitude.iloc[::max(1, len(track)//200)], track.latitude.iloc[::max(1, len(track)//200)],
               s=7, color="#1f77b4")
    ax.set(title=f"{partition} trip {trip_id}: GPS and reference route", xlabel="Longitude", ylabel="Latitude")
    ax.legend(["GPS trajectory", "Reference route"])
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    p = MAP_DIR / f"{partition}_{trip_id}_trajectory.png"
    fig.savefig(p, dpi=160); plt.close(fig); out.append(p)

    fig, ax = plt.subplots(figsize=(9, 7))
    lat_pad = max(0.002, float(track.latitude.max() - track.latitude.min()) * 0.08)
    lon_pad = max(0.002, float(track.longitude.max() - track.longitude.min()) * 0.08)
    bounds = (float(track.longitude.min()) - lon_pad, float(track.longitude.max()) + lon_pad,
              float(track.latitude.min()) - lat_pad, float(track.latitude.max()) + lat_pad)
    net_start, net_end = _network_lines_in_bounds(nodes, arcs, bounds, limit=12000)
    for p1, p2 in zip(net_start, net_end):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#777777", linewidth=.35, alpha=.45)
    ax.plot(track.longitude, track.latitude, color="#1f77b4", linewidth=1.0, label="GPS")
    cpath = CANDIDATE_DIR / f"{partition}_candidates_r{radius}.csv"
    if cpath.exists():
        cparts = []
        for chunk in pd.read_csv(cpath, dtype={"trip_id": str}, chunksize=250_000):
            selected = chunk.loc[chunk.trip_id == trip_id]
            if not selected.empty: cparts.append(selected)
        cdf = pd.concat(cparts, ignore_index=True) if cparts else pd.DataFrame()
        if not cdf.empty:
            cdf = cdf.nsmallest(500, "perpendicular_distance")
            for s, e in _arc_coordinates(nodes, arcs, cdf.candidate_arc_id.tolist()):
                ax.plot([s[1], e[1]], [s[0], e[0]], color="#2ca02c", linewidth=.65, alpha=.45)
    for s, e in _arc_coordinates(nodes, arcs, d["route"].arc_id.tolist()):
        ax.plot([s[1], e[1]], [s[0], e[0]], color="#d62728", linewidth=1.2, alpha=.8)
    ax.set(title=f"{partition} trip {trip_id}: road network and GPS", xlabel="Longitude", ylabel="Latitude")
    legend_handles = [
        Line2D([0], [0], color="#777777", lw=1, label="Road network"),
        Line2D([0], [0], color="#1f77b4", lw=1, label="GPS trajectory"),
        Line2D([0], [0], color="#d62728", lw=1.5, label="Reference route"),
        Line2D([0], [0], color="#2ca02c", lw=1, label="Candidate segments"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim"); fig.tight_layout()
    p = MAP_DIR / f"{partition}_{trip_id}_network.png"
    fig.savefig(p, dpi=160); plt.close(fig); out.append(p)
    return out


def candidate_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare existing candidate datasets and coverage at the observation level.

    Coverage here means an observation has >=1 generated candidate. True
    reference-route candidate coverage requires route-to-GPS alignment and is
    deliberately not inferred from proximity-only stats.
    """
    stats_parts = []
    quality_rows = []
    baseline_schema = None
    for path in sorted(CANDIDATE_DIR.glob("*_candidates_r*.csv")):
        name = path.stem
        prefix, _, radius_token = name.rpartition("_r")
        partition = prefix.removesuffix("_candidates")
        radius = int(radius_token)
        columns = list(pd.read_csv(path, nrows=0).columns)
        if baseline_schema is None:
            baseline_schema = columns
        row_count, missing = 0, {column: 0 for column in columns}
        duplicate_keys = 0
        key_cols = [col for col in ["trip_id", "observation_id", "candidate_arc_id"] if col in columns]
        for chunk in pd.read_csv(path, chunksize=250_000):
            row_count += len(chunk)
            for column in columns:
                missing[column] += int(chunk[column].isna().sum())
            if key_cols:
                duplicate_keys += int(chunk.duplicated(key_cols).sum())
        quality_rows.append({
            "file": path.name, "partition": partition, "radius_m": radius, "rows": row_count,
            "schema_matches_first_file": columns == baseline_schema,
            "duplicate_candidate_keys_within_chunks": duplicate_keys,
            "missing_values_total": int(sum(missing.values())),
            "missing_by_column": ";".join(f"{c}:{n}" for c, n in missing.items() if n),
        })
        stats_path = REPORT_DIR / f"{partition}_candidate_stats_r{radius}.csv"
        if stats_path.exists():
            s = pd.read_csv(stats_path, dtype={"trip_id": str})
            s["observation_candidate_coverage"] = s.observations_with_candidates / s.gps_observations
            s["candidate_miss_rate"] = s.zero_candidate_observations / s.gps_observations
            s["radius_m"] = radius
            stats_parts.append(s)
    detail = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()
    summary = (detail.groupby(["partition", "radius_m"], as_index=False)
               .agg(trips=("trip_id", "nunique"), gps_observations=("gps_observations", "sum"),
                    observations_with_candidates=("observations_with_candidates", "sum"),
                    zero_candidate_observations=("zero_candidate_observations", "sum"),
                    candidate_rows=("candidate_rows", "sum"))) if not detail.empty else pd.DataFrame()
    if not summary.empty:
        summary["observation_candidate_coverage"] = summary.observations_with_candidates / summary.gps_observations
        summary["candidate_miss_rate"] = summary.zero_candidate_observations / summary.gps_observations
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(REPORT_DIR / "candidate_diagnostics_by_trip.csv", index=False)
    summary.to_csv(REPORT_DIR / "candidate_diagnostics_summary.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(REPORT_DIR / "candidate_data_quality.csv", index=False)
    return detail, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["train", "student_test"], default="train")
    parser.add_argument("--trip-id", default="00000001")
    parser.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    args = parser.parse_args()
    print("Saved:", create_trip_map(args.partition, args.trip_id, args.radius))
    print("Saved:", *save_static_trip_plots(args.partition, args.trip_id, args.radius))
    _, summary = candidate_diagnostics()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

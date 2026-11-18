"""
PathSnap - Interactive Map Visualisation (EDA)

Creates interactive Folium maps for:
1. GPS trajectories (coloured by speed)
2. Road networks (arcs around the trip)
3. Candidate road segments

No reference / ground-truth route is drawn on any of these maps.

Examples:
    python -m src.maps --partition train --trip 00000001 --map-type all
    python -m src.maps --partition student_test --trip 00000007 --map-type gps
"""

import argparse

import folium
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen

from src.preprocessing import (
    PROJECT_ROOT,
    load_trip,
    preprocess_track,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maps"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette shared by every map.
GPS_COLOR = "#2563eb"
NETWORK_COLOR = "#8792a6"
CANDIDATE_COLOR = "#f59e0b"
START_COLOR = "#16a34a"
END_COLOR = "#111827"

# Esri tiles need no API key and load from a local file:// page.
# (OpenStreetMap blocks file:// pages; CARTO basemaps now need an API key.)
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, &copy; OpenStreetMap contributors"


# ============================================================
# SHARED HELPERS
# ============================================================

OSM_DE = "https://tile.openstreetmap.de/{z}/{x}/{y}.png"
OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'


def add_basemaps(map_object):
    """OpenStreetMap style by default (same look as the evaluator UI), Esri maps as options.

    Served from openstreetmap.de: tile.openstreetmap.org refuses pages opened from disk
    (file://), which shows an "Access blocked" grid in a double-clicked map.
    """
    folium.TileLayer(tiles=ESRI.format("World_Street_Map"), attr=ESRI_ATTR,
                     name="Esri street map", max_zoom=19, show=False).add_to(map_object)
    folium.TileLayer(tiles=ESRI.format("Canvas/World_Light_Gray_Base"), attr=ESRI_ATTR,
                     name="Light grey map", max_zoom=16, show=False).add_to(map_object)
    folium.TileLayer(tiles=OSM_DE, attr=OSM_ATTR, name="OpenStreetMap",
                     max_zoom=18, show=True).add_to(map_object)


def create_base_map(latitude, longitude, zoom=14):
    map_object = folium.Map(
        location=[float(latitude), float(longitude)],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )
    add_basemaps(map_object)

    Fullscreen(
        position="topleft",
        title="Full Screen",
        title_cancel="Exit Full Screen",
    ).add_to(map_object)
    return map_object


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def _track_summary(track):
    """Per-step speed (km/h) plus trip totals used in the info card."""
    lat = track["latitude"].to_numpy(dtype=float)
    lon = track["longitude"].to_numpy(dtype=float)
    t = track["timestamp"].to_numpy(dtype=float)
    step_m = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step_m / dt * 3.6, 0.0)
    speed = np.clip(np.nan_to_num(speed), 0, None)
    # 1 Hz GPS speed is noisy; a centred 9-step rolling median keeps the colouring readable.
    speed = pd.Series(speed).rolling(9, center=True, min_periods=1).median().to_numpy()
    return {
        "speed_kmh": speed,
        "length_km": float(step_m.sum() / 1000),
        "duration_min": float((t[-1] - t[0]) / 60) if len(t) > 1 else 0.0,
        "points": int(len(track)),
    }


def _bounds(track, pad_deg=0.002):
    return (
        float(track["latitude"].min()) - pad_deg,
        float(track["latitude"].max()) + pad_deg,
        float(track["longitude"].min()) - pad_deg,
        float(track["longitude"].max()) + pad_deg,
    )


def _arc_segments(nodes, arcs, arc_ids=None, bounds=None):
    """[[lat, lon], [lat, lon]] per valid arc; optionally only arcs touching `bounds`."""
    ends = arcs[["from_node_id", "to_node_id"]].apply(pd.to_numeric, errors="coerce")
    if arc_ids is not None:
        ids = np.asarray([a for a in arc_ids if 0 <= a < len(arcs)], dtype=int)
        ends = ends.iloc[ids]
    ends = ends.dropna()
    ends = ends[ends.from_node_id.between(0, len(nodes) - 1) & ends.to_node_id.between(0, len(nodes) - 1)]
    ll = nodes[["latitude", "longitude"]].to_numpy(dtype=float)
    a = ll[ends.from_node_id.astype(int).to_numpy()]
    b = ll[ends.to_node_id.astype(int).to_numpy()]
    ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    if bounds is not None:
        lat_lo, lat_hi, lon_lo, lon_hi = bounds

        def inside(p):
            return (p[:, 0] >= lat_lo) & (p[:, 0] <= lat_hi) & (p[:, 1] >= lon_lo) & (p[:, 1] <= lon_hi)

        ok &= inside(a) | inside(b)
    return [[a_.tolist(), b_.tolist()] for a_, b_ in zip(a[ok], b[ok])]


def _add_gps(map_object, track, summary, by_speed=True, weight=4.0):
    """GPS line with a white casing, coloured by speed, plus start/end markers."""
    coords = track[["latitude", "longitude"]].to_numpy(dtype=float).tolist()
    layer = folium.FeatureGroup(name="GPS trajectory", show=True)

    # White casing underneath keeps the line readable on any basemap.
    folium.PolyLine(coords, color="white", weight=weight + 3, opacity=0.9).add_to(layer)

    folium.PolyLine(coords, color=GPS_COLOR, weight=weight, opacity=0.95,
                    tooltip="GPS trajectory").add_to(layer)

    if by_speed and len(coords) > 2:
        # Optional layer: the same track coloured by speed (switch on in the layer control).
        speed = summary["speed_kmh"]
        vmax = float(max(10.0, np.percentile(speed, 98)))
        cmap = LinearColormap(["#1e3a8a", "#2563eb", "#06b6d4", "#facc15", "#ef4444"], vmin=0, vmax=vmax)
        speed_layer = folium.FeatureGroup(name=f"GPS speed (blue slow → red {vmax:.0f} km/h)", show=False)
        folium.ColorLine(coords, colors=np.clip(speed, 0, vmax).tolist(), colormap=cmap,
                         weight=weight, opacity=1).add_to(speed_layer)
        speed_layer.add_to(map_object)

    # Invisible hover targets so individual fixes can be inspected without hiding the line.
    step = max(1, len(coords) // 250)
    t0 = float(track["timestamp"].iloc[0])
    speed = summary["speed_kmh"]
    for i in range(0, len(coords), step):
        v = speed[min(i, len(speed) - 1)] if len(speed) else 0.0
        folium.CircleMarker(
            coords[i], radius=6, stroke=False, fill=True, fill_opacity=0,
            tooltip=f"Fix {i} · t = {float(track['timestamp'].iloc[i]) - t0:.0f} s · {v:.0f} km/h",
        ).add_to(layer)
    layer.add_to(map_object)

    ends = folium.FeatureGroup(name="Start / end", show=True)
    for point, color, label in ((coords[0], START_COLOR, "Start"), (coords[-1], END_COLOR, "End")):
        folium.CircleMarker(point, radius=9, color="white", weight=3, fill=True,
                            fill_color=color, fill_opacity=1, tooltip=label).add_to(ends)
    ends.add_to(map_object)
    return coords


def _add_info_card(map_object, title, partition, trip_id, summary, legend_rows, note=None):
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px">'
        f'<span style="display:inline-block;width:{w}px;height:{h}px;background:{c};'
        f'border-radius:{r};{extra}"></span>{label}</div>'
        for label, c, w, h, r, extra in legend_rows
    )
    stats = (
        f'{summary["points"]:,} GPS fixes · {summary["duration_min"]:.1f} min · '
        f'{summary["length_km"]:.2f} km'
    )
    note_html = f'<div style="margin-top:8px;color:#64748b;font-size:11px">{note}</div>' if note else ""
    html = (
        '<div style="position:fixed;bottom:28px;left:16px;z-index:9999;background:rgba(255,255,255,.96);'
        'padding:12px 14px;border-radius:10px;box-shadow:0 2px 10px rgba(15,23,42,.18);'
        'font:13px/1.35 \'Segoe UI\',system-ui,sans-serif;color:#0f172a;max-width:300px">'
        f'<div style="font-weight:700;font-size:14px">{title}</div>'
        f'<div style="color:#475569;margin:2px 0 8px">Trip {trip_id} · {partition.replace("_", "-")}</div>'
        f'<div style="color:#334155;margin-bottom:6px">{stats}</div>'
        f'{rows}{note_html}</div>'
    )
    map_object.get_root().html.add_child(folium.Element(html))


LEGEND_START = ("Start", START_COLOR, 12, 12, "50%", "border:2px solid white;box-shadow:0 0 0 1px #cbd5e1")
LEGEND_END = ("End", END_COLOR, 12, 12, "50%", "border:2px solid white;box-shadow:0 0 0 1px #cbd5e1")
LEGEND_NETWORK = ("Road network", NETWORK_COLOR, 22, 2, "1px", "")
LEGEND_CANDIDATE = ("Candidate road segments", CANDIDATE_COLOR, 22, 7, "3px", "")


def _load_clean_track(partition, trip_id):
    data = load_trip(partition, trip_id)
    track, _ = preprocess_track(data["track"])
    if track.empty:
        raise ValueError("No valid GPS observations found.")
    return data, track


def _finish(map_object, coords, path, *extra_lines):
    folium.LayerControl(collapsed=False, position="topright").add_to(map_object)
    map_object.fit_bounds(coords, padding=(30, 30))
    map_object.save(str(path))
    print(*extra_lines, sep="\n")
    return path


# ============================================================
# GPS TRAJECTORY MAP
# ============================================================

def create_gps_map(partition, trip_id):
    data, track = _load_clean_track(partition, trip_id)
    summary = _track_summary(track)
    map_object = create_base_map(track["latitude"].mean(), track["longitude"].mean())

    # Road network around the trip as grey context, like the evaluator UI.
    segments = _arc_segments(data["nodes"], data["arcs"], bounds=_bounds(track))
    net = folium.FeatureGroup(name="Road network", show=True)
    if segments:
        folium.PolyLine(segments, color=NETWORK_COLOR, weight=1.5, opacity=0.7).add_to(net)
    net.add_to(map_object)

    coords = _add_gps(map_object, track, summary, by_speed=True, weight=5)
    _add_info_card(map_object, "GPS trajectory", partition, trip_id, summary,
                   [LEGEND_NETWORK, ("GPS trajectory", GPS_COLOR, 22, 5, "3px", ""), LEGEND_START, LEGEND_END],
                   note="Hover the line for a fix's time and speed; switch on \"GPS speed\" in the layer menu to colour it by speed.")

    path = OUTPUT_DIR / f"{trip_id}_gps.html"
    return _finish(map_object, coords, path,
                   f"GPS map saved: {path}", f"GPS observations: {len(coords)}")


# ============================================================
# ROAD NETWORK MAP
# ============================================================

def create_road_network_map(partition, trip_id):
    data, track = _load_clean_track(partition, trip_id)
    nodes, arcs = data["nodes"], data["arcs"]
    if nodes.empty:
        raise ValueError("No road-network nodes available.")
    summary = _track_summary(track)
    map_object = create_base_map(track["latitude"].mean(), track["longitude"].mean(), zoom=13)

    # Only arcs around the trip: the full network can be 300,000 arcs (a 25 MB page).
    segments = _arc_segments(nodes, arcs, bounds=_bounds(track))
    net = folium.FeatureGroup(name=f"Road network ({len(segments):,} arcs near the trip)", show=True)
    if segments:
        folium.PolyLine(segments, color=NETWORK_COLOR, weight=1.6, opacity=0.85).add_to(net)
    net.add_to(map_object)

    coords = _add_gps(map_object, track, summary, by_speed=False)
    _add_info_card(map_object, "Road network around the trip", partition, trip_id, summary,
                   [LEGEND_NETWORK, ("GPS trajectory", GPS_COLOR, 22, 4, "2px", ""), LEGEND_START, LEGEND_END],
                   note=f"{len(segments):,} of {len(arcs):,} arcs in this trip's network lie in the box around the track (+~200 m).")

    path = OUTPUT_DIR / f"{trip_id}_road_network.html"
    return _finish(map_object, coords, path,
                   f"Road network map saved: {path}", f"Road segments displayed: {len(segments)} of {len(arcs)}")


# ============================================================
# CANDIDATE ROAD SEGMENT MAP
# ============================================================

def create_candidate_map(partition, trip_id, radius=50, max_display=1000):
    data, track = _load_clean_track(partition, trip_id)
    nodes, arcs = data["nodes"], data["arcs"]
    if nodes.empty or arcs.empty:
        raise ValueError("Road network nodes or arcs are unavailable.")

    candidate_path = PROCESSED_DIR / f"{partition}_candidates_r{radius}.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    chunks = [c[c["trip_id"] == str(trip_id)]
              for c in pd.read_csv(candidate_path, dtype={"trip_id": str},
                                   usecols=["trip_id", "observation_id", "candidate_arc_id"],
                                   chunksize=250_000)]
    candidates = pd.concat(chunks, ignore_index=True)
    if candidates.empty:
        raise ValueError("No candidates found for this trip.")
    n_obs = candidates["observation_id"].nunique()
    shown_obs = candidates["observation_id"].drop_duplicates().head(max_display)
    candidates = candidates[candidates["observation_id"].isin(shown_obs)]
    arc_ids = pd.to_numeric(candidates["candidate_arc_id"], errors="coerce").dropna().astype(int).unique()

    summary = _track_summary(track)
    map_object = create_base_map(track["latitude"].mean(), track["longitude"].mean())

    context = _arc_segments(nodes, arcs, bounds=_bounds(track))
    net = folium.FeatureGroup(name="Road network (context)", show=True)
    if context:
        folium.PolyLine(context, color=NETWORK_COLOR, weight=1, opacity=0.45).add_to(net)
    net.add_to(map_object)

    segments = _arc_segments(nodes, arcs, arc_ids=arc_ids)
    cand = folium.FeatureGroup(name=f"Candidate segments (r = {radius} m)", show=True)
    if segments:
        folium.PolyLine(segments, color=CANDIDATE_COLOR, weight=8, opacity=0.8,
                        tooltip=f"Candidate arc within {radius} m").add_to(cand)
    cand.add_to(map_object)

    # Thin GPS line so the candidate arcs underneath stay visible.
    coords = _add_gps(map_object, track, summary, by_speed=False, weight=1.8)
    shown_note = (f"first {len(shown_obs):,} of {n_obs:,} fixes" if n_obs > len(shown_obs)
                  else f"all {n_obs:,} fixes")
    _add_info_card(map_object, f"Candidate road segments (r = {radius} m)", partition, trip_id, summary,
                   [LEGEND_CANDIDATE, LEGEND_NETWORK, ("GPS trajectory", GPS_COLOR, 22, 4, "2px", ""),
                    LEGEND_START, LEGEND_END],
                   note=f"{len(segments):,} distinct candidate arcs shown for {shown_note}.")

    path = OUTPUT_DIR / f"{trip_id}_candidates_r{radius}.html"
    return _finish(map_object, coords, path,
                   f"Candidate map saved: {path}", f"Candidate arcs displayed: {len(segments)}")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PathSnap interactive map visualisation")
    parser.add_argument("--partition", choices=["train", "student_test"], default="train")
    parser.add_argument("--trip", default="00000001")
    parser.add_argument("--radius", type=int, choices=[30, 50, 75], default=50)
    parser.add_argument("--map-type", choices=["gps", "network", "candidates", "all"], default="all")
    args = parser.parse_args()

    if args.map_type in ["gps", "all"]:
        create_gps_map(args.partition, args.trip)
    if args.map_type in ["network", "all"]:
        create_road_network_map(args.partition, args.trip)
    if args.map_type in ["candidates", "all"]:
        create_candidate_map(args.partition, args.trip, args.radius)


if __name__ == "__main__":
    main()


from src.preprocessing import load_trip
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

trips = {
    "train": {
        "00000002": [53],
        "00000019": [94, 151, 300],
        "00000033": [164],
        "00000041": [12],
        "00000042": [1212],
        "00000069": [153],
        "00000075": [31, 62],
        "00000080": [118],
        "00000086": [1],
        "00000089": [192],
    },
    "student_test": {
        "00000076": [12]
    }
}

transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:32632", always_xy=True
)

results = []

for partition, trip_data in trips.items():
    for trip_id, positions in trip_data.items():

        data = load_trip(partition, trip_id)

        nodes = data["nodes"]
        arcs = data["arcs"]
        route = data["route"]
        track = data["track"]

        nx, ny = transformer.transform(
            nodes["longitude"].to_numpy(),
            nodes["latitude"].to_numpy()
        )

        tx, ty = transformer.transform(
            track["longitude"].to_numpy(),
            track["latitude"].to_numpy()
        )

        gps_tree = cKDTree(np.column_stack((tx, ty)))

        edges = {}

        for _, arc in arcs.iterrows():
            u = arc["from_node_id"]
            v = arc["to_node_id"]

            if not np.isfinite(u) or not np.isfinite(v):
                continue

            if u % 1 != 0 or v % 1 != 0:
                continue

            u, v = int(u), int(v)

            if not (0 <= u < len(nodes) and 0 <= v < len(nodes)):
                continue

            length = np.hypot(nx[v] - nx[u], ny[v] - ny[u])
            edges[(u, v)] = length

        rows = [edge[0] for edge in edges]
        cols = [edge[1] for edge in edges]
        weights = list(edges.values())

        graph = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(nodes), len(nodes))
        )

        for i in positions:

            arc_a = arcs.iloc[int(route.iloc[i]["arc_id"])]
            arc_b = arcs.iloc[int(route.iloc[i + 1]["arc_id"])]

            source = int(arc_a["to_node_id"])
            target = int(arc_b["from_node_id"])

            distances, predecessors = dijkstra(
                graph,
                directed=True,
                indices=source,
                return_predecessors=True
            )

            if not np.isfinite(distances[target]):
                print(f"\n{trip_id}, position {i}: NO PATH")
                continue

            path = [target]
            current = target

            while current != source:
                current = predecessors[current]
                if current < 0:
                    break
                path.append(int(current))

            path.reverse()

            path_x = nx[path]
            path_y = ny[path]

            path_tree = cKDTree(
                np.column_stack((path_x, path_y))
            )

            gps_distances, _ = path_tree.query(
                np.column_stack((tx, ty)),
                k=1
            )

            nearest_gps_distance = float(gps_distances.min())
            gps_points_near_path = int(
                np.sum(gps_distances <= 30)
            )

            results.append({
                "partition": partition,
                "trip_id": trip_id,
                "route_position": i,
                "path_distance_m": float(distances[target]),
                "path_edges": len(path) - 1,
                "nearest_gps_distance_m": nearest_gps_distance,
                "gps_points_within_30m": gps_points_near_path
            })

            print(f"\nTrip: {trip_id}")
            print(f"Route position: {i}")
            print(f"Connecting path distance: {distances[target]:.2f} m")
            print(f"Connecting path edges: {len(path) - 1}")
            print(f"Nearest GPS observation to path: {nearest_gps_distance:.2f} m")
            print(f"GPS observations within 30 m: {gps_points_near_path}")

if results:
    pd.DataFrame(results).to_csv(
        "outputs/reports/route_gap_gps_diagnostics.csv",
        index=False
    )
    print("\nReport saved to outputs/reports/route_gap_gps_diagnostics.csv")

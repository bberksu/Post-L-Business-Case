from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

try:
    import pulp
except ImportError as exc:
    raise SystemExit("Please install pulp: pip install pulp") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import geopandas as gpd
    import contextily as cx
    from shapely.geometry import Point
except ImportError:
    gpd = None
    cx = None
    Point = None


DATA_PATH = Path("/Users/macbook/Desktop/logistics case elab/data_Maastricht_2025.xlsx")
OUTPUT_DIR = Path("/Users/macbook/Desktop/mip")


# Costs from the case / assumptions
FIXED_COST_SERVICE_POINT = 50_000
FIXED_COST_APL = 15_000
DELIVERY_COST_PER_PACKAGE_KM = 1.50
STORAGE_COST_PER_PACKAGE_DAY = 0.10
AVERAGE_PICKUP_STORAGE_DAYS = 2.0

# APL assumptions
APL_CAPACITY_DAILY = 90
APL_SERVICE_COST_PER_PICKUP = 0.25
APL_REPLENISHMENT_COST_PER_PACKAGE_KM = 0.40

# Capacity assumptions
DEFAULT_SP_CAPACITY_DAILY = 350

# Network assumptions
N_APL_CANDIDATES = 20
MIN_DISTANCE_BETWEEN_APL_CANDIDATES_KM = 0.8
MAX_ASSIGNMENT_DISTANCE_KM = 8.0

# Optional constraints
MIN_OPEN_SERVICE_POINTS = None
MIN_OPEN_LOCATIONS = None
MAX_OPEN_LOCATIONS = None
MAX_OPEN_APLS = None
PROTECTED_TOP_SP_COUNT = 0

# Penalty for long access distance
LONG_DISTANCE_PENALTY_PER_PACKAGE_KM = 0.25

# Solver settings
SOLVER_TIME_LIMIT_SECONDS = 600
SOLVER_RELATIVE_GAP = 0.01


@dataclass
class Facility:
    facility_id: str
    facility_type: str  # "SP" or "APL"
    x: float
    y: float
    fixed_cost: float
    capacity_daily: float


def load_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sp = pd.read_excel(path, sheet_name="Service Point Locations")
    activity = pd.read_excel(path, sheet_name="Daily Activity")
    cbs = pd.read_excel(path, sheet_name="CBS Squares")
    nodes = pd.read_excel(path, sheet_name="Nodes")
    edges = pd.read_excel(path, sheet_name="Edges")
    return sp, activity, cbs, nodes, edges


def distance_km(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * 0.1 / 1000


def estimate_cbs_demand(cbs: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    cbs_known = cbs.dropna(subset=["Population", "X", "Y"]).copy()

    total_annual_demand = (activity["Deliveries"] + activity["Pickups"]).sum()
    total_population = cbs_known["Population"].sum()

    cbs_known["demand_weight"] = cbs_known["Population"] / total_population
    cbs_known["annual_demand"] = cbs_known["demand_weight"] * total_annual_demand

    return cbs_known.reset_index(drop=True)


def build_existing_facilities(sp: pd.DataFrame, activity: pd.DataFrame) -> List[Facility]:
    activity_by_sp = (
        activity.groupby("Location ID", as_index=False)
        .agg(actual_pickups=("Pickups", "sum"))
    )
    activity_by_sp["avg_daily_pickups"] = activity_by_sp["actual_pickups"] / 365

    sp2 = sp.merge(activity_by_sp, on="Location ID", how="left")
    sp2["avg_daily_pickups"] = sp2["avg_daily_pickups"].fillna(0)

    facilities = []
    for _, row in sp2.iterrows():
        capacity = max(DEFAULT_SP_CAPACITY_DAILY, 1.35 * row["avg_daily_pickups"])

        facilities.append(
            Facility(
                facility_id=f"SP_{int(row['Location ID'])}",
                facility_type="SP",
                x=float(row["X"]),
                y=float(row["Y"]),
                fixed_cost=FIXED_COST_SERVICE_POINT,
                capacity_daily=float(capacity),
            )
        )

    return facilities


def protected_service_points(activity: pd.DataFrame) -> set[str]:
    if PROTECTED_TOP_SP_COUNT <= 0:
        return set()

    top_sp = (
        activity.assign(total=activity["Deliveries"] + activity["Pickups"])
        .groupby("Location ID", as_index=False)
        .agg(actual_total=("total", "sum"))
        .sort_values("actual_total", ascending=False)
        .head(PROTECTED_TOP_SP_COUNT)
    )

    return {f"SP_{int(location_id)}" for location_id in top_sp["Location ID"]}


def generate_apl_candidates(cbs_demand: pd.DataFrame, existing_facilities: List[Facility]) -> List[Facility]:
    cbs_x = cbs_demand["X"].to_numpy()
    cbs_y = cbs_demand["Y"].to_numpy()

    sp_x = np.array([facility.x for facility in existing_facilities])
    sp_y = np.array([facility.y for facility in existing_facilities])

    nearest_dist = distance_km(
        cbs_x[:, None],
        cbs_y[:, None],
        sp_x[None, :],
        sp_y[None, :],
    ).min(axis=1)

    candidates = cbs_demand.copy()
    candidates["nearest_sp_dist_km"] = nearest_dist
    candidates["apl_score"] = candidates["Population"] * candidates["nearest_sp_dist_km"]
    candidates = candidates[candidates["Population"] > 0].copy()
    candidates = candidates.sort_values("apl_score", ascending=False).reset_index(drop=True)

    selected_rows = []

    for _, row in candidates.iterrows():
        if len(selected_rows) >= N_APL_CANDIDATES:
            break

        if not selected_rows:
            selected_rows.append(row)
            continue

        selected_x = np.array([selected["X"] for selected in selected_rows])
        selected_y = np.array([selected["Y"] for selected in selected_rows])

        dist_to_selected = distance_km(
            np.array([row["X"]]),
            np.array([row["Y"]]),
            selected_x,
            selected_y,
        )

        if dist_to_selected.min() >= MIN_DISTANCE_BETWEEN_APL_CANDIDATES_KM:
            selected_rows.append(row)

    if len(selected_rows) < N_APL_CANDIDATES:
        selected_squares = {row["Square"] for row in selected_rows}

        for _, row in candidates.iterrows():
            if len(selected_rows) >= N_APL_CANDIDATES:
                break

            if row["Square"] not in selected_squares:
                selected_rows.append(row)
                selected_squares.add(row["Square"])

    apls = []
    for index, row in enumerate(selected_rows, start=1):
        apls.append(
            Facility(
                facility_id=f"APL_{index}_{row['Square']}",
                facility_type="APL",
                x=float(row["X"]),
                y=float(row["Y"]),
                fixed_cost=FIXED_COST_APL,
                capacity_daily=APL_CAPACITY_DAILY,
            )
        )

    return apls


def build_road_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[csr_matrix, dict[int, int], np.ndarray]:
    node_ids = nodes["NODE ID"].astype(int).to_numpy()
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    row_idx = []
    col_idx = []
    weights = []

    for _, edge in edges.iterrows():
        v1 = int(edge["V1"])
        v2 = int(edge["V2"])

        if v1 not in node_id_to_idx or v2 not in node_id_to_idx:
            continue

        i = node_id_to_idx[v1]
        j = node_id_to_idx[v2]
        dist_km_value = float(edge["DIST"]) / 1000

        row_idx.append(i)
        col_idx.append(j)
        weights.append(dist_km_value)

        one_way = edge.get("ONE_WAY", 0)
        if pd.isna(one_way):
            one_way = 0

        if int(one_way) == 0:
            row_idx.append(j)
            col_idx.append(i)
            weights.append(dist_km_value)

    graph = csr_matrix((weights, (row_idx, col_idx)), shape=(len(nodes), len(nodes)))
    node_xy = nodes[["X", "Y"]].to_numpy(dtype=float)

    return graph, node_id_to_idx, node_xy


def nearest_node_indices(points_x: np.ndarray, points_y: np.ndarray, node_xy: np.ndarray) -> np.ndarray:
    tree = cKDTree(node_xy)
    points = np.column_stack([points_x, points_y])
    _, nearest_indices = tree.query(points)
    return nearest_indices


def build_straight_line_distance_matrix(cbs_demand: pd.DataFrame, facilities: List[Facility]) -> np.ndarray:
    cbs_x = cbs_demand["X"].to_numpy()
    cbs_y = cbs_demand["Y"].to_numpy()
    fac_x = np.array([facility.x for facility in facilities])
    fac_y = np.array([facility.y for facility in facilities])

    return distance_km(
        cbs_x[:, None],
        cbs_y[:, None],
        fac_x[None, :],
        fac_y[None, :],
    )


def build_road_distance_matrix(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> np.ndarray:
    graph, _, node_xy = build_road_graph(nodes, edges)

    cbs_node_idx = nearest_node_indices(
        cbs_demand["X"].to_numpy(),
        cbs_demand["Y"].to_numpy(),
        node_xy,
    )

    facility_x = np.array([facility.x for facility in facilities])
    facility_y = np.array([facility.y for facility in facilities])

    facility_node_idx = nearest_node_indices(facility_x, facility_y, node_xy)

    unique_cbs_nodes, inverse_cbs = np.unique(cbs_node_idx, return_inverse=True)

    shortest_paths = dijkstra(
        csgraph=graph,
        directed=True,
        indices=unique_cbs_nodes,
    )

    road_dist_matrix = shortest_paths[inverse_cbs][:, facility_node_idx]
    fallback_dist_matrix = build_straight_line_distance_matrix(cbs_demand, facilities)

    return np.where(np.isfinite(road_dist_matrix), road_dist_matrix, fallback_dist_matrix)


def raw_pickup_probability(dist_km: np.ndarray) -> np.ndarray:
    return np.select(
        [dist_km <= 0.5, dist_km <= 1.0, dist_km <= 2.0],
        [0.80, 0.65, 0.45],
        default=0.25,
    )


def calibrate_pickup_scale(
    cbs_demand: pd.DataFrame,
    activity: pd.DataFrame,
    dist_matrix: np.ndarray,
    n_existing_facilities: int,
) -> float:
    demand = cbs_demand["annual_demand"].to_numpy()

    nearest_dist = dist_matrix[:, :n_existing_facilities].min(axis=1)
    raw_prob = raw_pickup_probability(nearest_dist)
    estimated_raw_share = np.average(raw_prob, weights=demand)

    actual_total = (activity["Deliveries"] + activity["Pickups"]).sum()
    actual_pickup_share = activity["Pickups"].sum() / actual_total

    if estimated_raw_share <= 0:
        return 1.0

    return float(actual_pickup_share / estimated_raw_share)


def build_cost_matrices(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
    pickup_probability_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    demand = cbs_demand["annual_demand"].to_numpy()

    p_pick = np.clip(
        raw_pickup_probability(dist_matrix) * pickup_probability_scale,
        0.05,
        0.95,
    )

    annual_pickups = demand[:, None] * p_pick
    annual_deliveries = demand[:, None] * (1 - p_pick)

    assignment_cost = (
        annual_deliveries * dist_matrix * DELIVERY_COST_PER_PACKAGE_KM
        + annual_pickups * STORAGE_COST_PER_PACKAGE_DAY * AVERAGE_PICKUP_STORAGE_DAYS
        + demand[:, None] * np.maximum(dist_matrix - 2.0, 0) * LONG_DISTANCE_PENALTY_PER_PACKAGE_KM
    )

    for j, facility in enumerate(facilities):
        if facility.facility_type == "APL":
            assignment_cost[:, j] += annual_pickups[:, j] * APL_SERVICE_COST_PER_PICKUP
            assignment_cost[:, j] += (
                annual_pickups[:, j]
                * dist_matrix[:, j]
                * APL_REPLENISHMENT_COST_PER_PACKAGE_KM
            )

    daily_pickups = annual_pickups / 365

    return assignment_cost, daily_pickups


def solve_mip(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
    assignment_cost: np.ndarray,
    daily_pickups: np.ndarray,
    protected_sp_ids: set[str],
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    n_cbs = len(cbs_demand)
    n_facilities = len(facilities)

    allowed_pairs = [
        (i, j)
        for i in range(n_cbs)
        for j in range(n_facilities)
        if dist_matrix[i, j] <= MAX_ASSIGNMENT_DISTANCE_KM
    ]

    problem = pulp.LpProblem("PostNL_Capacitated_Facility_Location", pulp.LpMinimize)

    y = {
        j: pulp.LpVariable(f"open_{j}_{facilities[j].facility_id}", lowBound=0, upBound=1, cat="Binary")
        for j in range(n_facilities)
    }

    x = {
        (i, j): pulp.LpVariable(f"assign_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
        for i, j in allowed_pairs
    }

    problem += (
        pulp.lpSum(facilities[j].fixed_cost * y[j] for j in range(n_facilities))
        + pulp.lpSum(assignment_cost[i, j] * x[(i, j)] for i, j in allowed_pairs)
    )

    for i in range(n_cbs):
        candidate_js = [j for j in range(n_facilities) if (i, j) in x]
        problem += pulp.lpSum(x[(i, j)] for j in candidate_js) == 1, f"assign_once_{i}"

    for i, j in allowed_pairs:
        problem += x[(i, j)] <= y[j], f"assign_only_if_open_{i}_{j}"

    for j, facility in enumerate(facilities):
        assigned_is = [i for i in range(n_cbs) if (i, j) in x]
        problem += (
            pulp.lpSum(daily_pickups[i, j] * x[(i, j)] for i in assigned_is)
            <= facility.capacity_daily * y[j]
        ), f"capacity_{j}"

    sp_indices = [j for j, facility in enumerate(facilities) if facility.facility_type == "SP"]
    apl_indices = [j for j, facility in enumerate(facilities) if facility.facility_type == "APL"]

    if MIN_OPEN_SERVICE_POINTS is not None:
        problem += pulp.lpSum(y[j] for j in sp_indices) >= MIN_OPEN_SERVICE_POINTS, "min_open_service_points"

    if MIN_OPEN_LOCATIONS is not None:
        problem += pulp.lpSum(y[j] for j in range(n_facilities)) >= MIN_OPEN_LOCATIONS, "min_open_locations"

    if MAX_OPEN_LOCATIONS is not None:
        problem += pulp.lpSum(y[j] for j in range(n_facilities)) <= MAX_OPEN_LOCATIONS, "max_open_locations"

    if MAX_OPEN_APLS is not None:
        problem += pulp.lpSum(y[j] for j in apl_indices) <= MAX_OPEN_APLS, "max_open_apls"

    for j, facility in enumerate(facilities):
        if facility.facility_id in protected_sp_ids:
            problem += y[j] == 1, f"protect_{facility.facility_id}"

    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=SOLVER_TIME_LIMIT_SECONDS,
        gapRel=SOLVER_RELATIVE_GAP,
    )

    problem.solve(solver)

    status = pulp.LpStatus[problem.status]
    objective_value = float(pulp.value(problem.objective))

    open_solution = np.array(
        [int(round(pulp.value(y[j]) or 0)) for j in range(n_facilities)],
        dtype=int,
    )

    assignment_solution = np.full(n_cbs, -1, dtype=int)

    for i in range(n_cbs):
        assigned_candidates = [
            (j, pulp.value(x[(i, j)]) or 0)
            for j in range(n_facilities)
            if (i, j) in x
        ]
        assignment_solution[i] = max(assigned_candidates, key=lambda item: item[1])[0]

    return open_solution, assignment_solution, status, objective_value


def evaluate_solution_components(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    open_solution: np.ndarray,
    assignment_solution: np.ndarray,
    dist_matrix: np.ndarray,
    pickup_probability_scale: float,
) -> dict:
    demand = cbs_demand["annual_demand"].to_numpy()
    assigned_dist = dist_matrix[np.arange(len(cbs_demand)), assignment_solution]

    p_pick = np.clip(
        raw_pickup_probability(assigned_dist) * pickup_probability_scale,
        0.05,
        0.95,
    )

    annual_pickups = demand * p_pick
    annual_deliveries = demand * (1 - p_pick)

    fixed_cost = sum(
        facility.fixed_cost
        for facility, is_open in zip(facilities, open_solution)
        if is_open
    )

    delivery_cost = float(np.sum(annual_deliveries * assigned_dist * DELIVERY_COST_PER_PACKAGE_KM))
    storage_cost = float(np.sum(annual_pickups * STORAGE_COST_PER_PACKAGE_DAY * AVERAGE_PICKUP_STORAGE_DAYS))

    long_distance_penalty = float(
        np.sum(demand * np.maximum(assigned_dist - 2.0, 0) * LONG_DISTANCE_PENALTY_PER_PACKAGE_KM)
    )

    facility_types = np.array([facility.facility_type for facility in facilities])
    assigned_is_apl = facility_types[assignment_solution] == "APL"

    apl_operating_cost = float(
        np.sum(annual_pickups[assigned_is_apl] * APL_SERVICE_COST_PER_PICKUP)
        + np.sum(
            annual_pickups[assigned_is_apl]
            * assigned_dist[assigned_is_apl]
            * APL_REPLENISHMENT_COST_PER_PACKAGE_KM
        )
    )

    pickup_by_facility_daily = np.zeros(len(facilities))
    np.add.at(pickup_by_facility_daily, assignment_solution, annual_pickups / 365)

    capacities = np.array([facility.capacity_daily for facility in facilities])
    overflow_daily = np.maximum(pickup_by_facility_daily - capacities, 0)

    total_pickups_daily = pickup_by_facility_daily.sum()
    city_bounce_rate = overflow_daily.sum() / total_pickups_daily if total_pickups_daily > 0 else 0

    location_bounce_rates = np.divide(
        overflow_daily,
        pickup_by_facility_daily,
        out=np.zeros_like(overflow_daily),
        where=pickup_by_facility_daily > 0,
    )

    total_cost = fixed_cost + delivery_cost + storage_cost + apl_operating_cost + long_distance_penalty

    population_within_1km = float(demand[assigned_dist <= 1.0].sum() / demand.sum())
    population_within_2km = float(demand[assigned_dist <= 2.0].sum() / demand.sum())
    population_over_3km = float(demand[assigned_dist > 3.0].sum() / demand.sum())

    return {
        "fixed_cost": float(fixed_cost),
        "delivery_cost": delivery_cost,
        "storage_cost": storage_cost,
        "apl_operating_cost": apl_operating_cost,
        "penalty_cost": long_distance_penalty,
        "total_cost": float(total_cost),
        "open_locations": int(open_solution.sum()),
        "open_service_points": int(sum(open_solution[j] for j, f in enumerate(facilities) if f.facility_type == "SP")),
        "open_apls": int(sum(open_solution[j] for j, f in enumerate(facilities) if f.facility_type == "APL")),
        "annual_pickups": float(annual_pickups.sum()),
        "annual_deliveries": float(annual_deliveries.sum()),
        "pickup_share": float(annual_pickups.sum() / demand.sum()),
        "avg_distance_km": float(np.average(assigned_dist, weights=demand)),
        "max_distance_km": float(assigned_dist.max()),
        "population_within_1km": population_within_1km,
        "population_within_2km": population_within_2km,
        "population_over_3km": population_over_3km,
        "city_bounce_rate": float(city_bounce_rate),
        "max_location_bounce_rate": float(location_bounce_rates.max()),
    }


def summarize_facilities(facilities: List[Facility], open_solution: np.ndarray) -> pd.DataFrame:
    rows = []

    for facility, is_open in zip(facilities, open_solution):
        rows.append(
            {
                "facility_id": facility.facility_id,
                "facility_type": facility.facility_type,
                "open": int(is_open),
                "x": facility.x,
                "y": facility.y,
                "fixed_cost": facility.fixed_cost,
                "capacity_daily": facility.capacity_daily,
            }
        )

    return pd.DataFrame(rows)


def plot_solution(cbs_demand: pd.DataFrame, facility_summary: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return

    open_sp = facility_summary[(facility_summary["facility_type"] == "SP") & (facility_summary["open"] == 1)]
    closed_sp = facility_summary[(facility_summary["facility_type"] == "SP") & (facility_summary["open"] == 0)]
    open_apl = facility_summary[(facility_summary["facility_type"] == "APL") & (facility_summary["open"] == 1)]

    plt.figure(figsize=(10, 7))

    plt.scatter(
        cbs_demand["X"],
        cbs_demand["Y"],
        s=np.clip(cbs_demand["Population"] / 12, 8, 160),
        color="#A0A7B0",
        alpha=0.28,
        label="CBS demand squares",
    )

    if not closed_sp.empty:
        plt.scatter(
            closed_sp["x"],
            closed_sp["y"],
            color="#737373",
            marker="x",
            s=65,
            label="Closed SP",
        )

    if not open_sp.empty:
        plt.scatter(
            open_sp["x"],
            open_sp["y"],
            color="#D7191C",
            edgecolor="white",
            linewidth=0.8,
            s=85,
            label="Open SP",
        )

    if not open_apl.empty:
        plt.scatter(
            open_apl["x"],
            open_apl["y"],
            color="#2F6F9F",
            edgecolor="white",
            linewidth=0.8,
            marker="^",
            s=110,
            label="Open APL",
        )

    plt.axis("equal")
    plt.grid(alpha=0.18)
    plt.title("MILP Capacitated Facility Location Solution")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mip_best_network.png", dpi=300)
    plt.close()


def convert_xy_to_rd(
    cbs: pd.DataFrame,
    x_values: pd.Series,
    y_values: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    cbs_coord = cbs.dropna(subset=["X", "Y", "Square"]).copy()

    cbs_coord["rd_x"] = (
        cbs_coord["Square"]
        .astype(str)
        .str.extract(r"E(\d+)")[0]
        .astype(float)
        * 100
    )

    cbs_coord["rd_y"] = (
        cbs_coord["Square"]
        .astype(str)
        .str.extract(r"N(\d+)")[0]
        .astype(float)
        * 100
    )

    model_x = np.polyfit(cbs_coord["X"], cbs_coord["rd_x"], 1)
    model_y = np.polyfit(cbs_coord["Y"], cbs_coord["rd_y"], 1)

    rd_x = model_x[0] * x_values + model_x[1]
    rd_y = model_y[0] * y_values + model_y[1]

    return rd_x.to_numpy(), rd_y.to_numpy()


def plot_current_vs_mip_real_map(
    sp: pd.DataFrame,
    cbs: pd.DataFrame,
    facility_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    if plt is None or gpd is None or cx is None or Point is None:
        print("geopandas/contextily/pyproj/matplotlib not installed; skipping real map plot.")
        return

    sp_map = sp.copy()
    sp_map["rd_x"], sp_map["rd_y"] = convert_xy_to_rd(cbs, sp_map["X"], sp_map["Y"])

    mip_map = facility_summary.copy()
    mip_map["rd_x"], mip_map["rd_y"] = convert_xy_to_rd(cbs, mip_map["x"], mip_map["y"])

    sp_gdf = gpd.GeoDataFrame(
        sp_map,
        geometry=[Point(x, y) for x, y in zip(sp_map["rd_x"], sp_map["rd_y"])],
        crs="EPSG:28992",
    ).to_crs(epsg=3857)

    mip_gdf = gpd.GeoDataFrame(
        mip_map,
        geometry=[Point(x, y) for x, y in zip(mip_map["rd_x"], mip_map["rd_y"])],
        crs="EPSG:28992",
    ).to_crs(epsg=3857)

    kept_sp = mip_gdf[(mip_gdf["facility_type"] == "SP") & (mip_gdf["open"] == 1)]
    closed_sp = mip_gdf[(mip_gdf["facility_type"] == "SP") & (mip_gdf["open"] == 0)]
    opened_apl = mip_gdf[(mip_gdf["facility_type"] == "APL") & (mip_gdf["open"] == 1)]

    minx, miny, maxx, maxy = mip_gdf.total_bounds
    x_buffer = 2500
    y_buffer = 2500

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sp_gdf.plot(
        ax=axes[0],
        color="#D7191C",
        markersize=45,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.95,
        label="Existing service point",
    )

    axes[0].set_title(
        "Current Network: 35 Service Points",
        fontsize=15,
        fontweight="bold",
    )

    if not closed_sp.empty:
        closed_sp.plot(
            ax=axes[1],
            color="#737373",
            markersize=45,
            marker="x",
            linewidth=1.3,
            alpha=0.85,
            label="Closed existing SP",
        )

    if not kept_sp.empty:
        kept_sp.plot(
            ax=axes[1],
            color="#D7191C",
            markersize=55,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            label="Kept existing SP",
        )

    if not opened_apl.empty:
        opened_apl.plot(
            ax=axes[1],
            color="#2F6F9F",
            markersize=75,
            marker="^",
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            label="Opened APL",
        )

    axes[1].set_title(
        f"MILP Optimized Network: {len(kept_sp)} SPs + {len(opened_apl)} APLs",
        fontsize=15,
        fontweight="bold",
    )

    for ax in axes:
        ax.set_xlim(minx - x_buffer, maxx + x_buffer)
        ax.set_ylim(miny - y_buffer, maxy + y_buffer)

        try:
            cx.add_basemap(
                ax,
                source=cx.providers.CartoDB.Positron,
                zoom=13,
            )
        except Exception as exc:
            print(f"Could not load basemap: {exc}")

        ax.set_axis_off()
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#D9E2EC",
            fontsize=10,
        )

    fig.suptitle(
        "Current Network vs MILP Solution",
        fontsize=20,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.035,
        "The MILP closes selected service points and opens APLs while satisfying capacity and bounce-rate constraints.",
        ha="center",
        fontsize=11,
        color="#4B5C70",
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08, wspace=0.04)
    fig.savefig(output_dir / "mip_current_vs_optimized_real_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    open_solution: np.ndarray,
    assignment_solution: np.ndarray,
    summary: dict,
    sp: pd.DataFrame,
    cbs: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    facility_summary = summarize_facilities(facilities, open_solution)
    facility_summary.to_csv(OUTPUT_DIR / "mip_best_solution_facilities.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "mip_best_solution_summary.csv", index=False)

    assignments = cbs_demand[["Square", "Population", "X", "Y", "annual_demand"]].copy()
    assignments["assigned_facility_id"] = [facilities[j].facility_id for j in assignment_solution]
    assignments["assigned_facility_type"] = [facilities[j].facility_type for j in assignment_solution]
    assignments.to_csv(OUTPUT_DIR / "mip_cbs_assignments.csv", index=False)

    plot_solution(cbs_demand, facility_summary, OUTPUT_DIR)
    plot_current_vs_mip_real_map(sp, cbs, facility_summary, OUTPUT_DIR)


def print_decisions(facility_summary: pd.DataFrame) -> None:
    open_sp = (
        facility_summary[(facility_summary["facility_type"] == "SP") & (facility_summary["open"] == 1)]
        ["facility_id"]
        .str.replace("SP_", "", regex=False)
        .tolist()
    )

    closed_sp = (
        facility_summary[(facility_summary["facility_type"] == "SP") & (facility_summary["open"] == 0)]
        ["facility_id"]
        .str.replace("SP_", "", regex=False)
        .tolist()
    )

    open_apl = facility_summary[
        (facility_summary["facility_type"] == "APL") & (facility_summary["open"] == 1)
    ]["facility_id"].tolist()

    closed_apl = facility_summary[
        (facility_summary["facility_type"] == "APL") & (facility_summary["open"] == 0)
    ]["facility_id"].tolist()

    print("\nNetwork decisions")
    print(f"Keep open existing SPs ({len(open_sp)}):")
    print(open_sp)

    print(f"\nClose existing SPs ({len(closed_sp)}):")
    print(closed_sp)

    print(f"\nOpen APLs ({len(open_apl)}):")
    print(open_apl)

    print(f"\nDo not open APL candidates ({len(closed_apl)}):")
    print(closed_apl)


def main() -> None:
    print("Loading data...")
    sp, activity, cbs, nodes, edges = load_data(DATA_PATH)

    print("Estimating demand by CBS square...")
    cbs_demand = estimate_cbs_demand(cbs, activity)

    print("Building candidate facilities...")
    existing_facilities = build_existing_facilities(sp, activity)
    apl_candidates = generate_apl_candidates(cbs_demand, existing_facilities)
    facilities = existing_facilities + apl_candidates

    protected_ids = protected_service_points(activity)

    print(f"Protected high-volume SPs: {sorted(protected_ids)}")
    print(f"Existing service points: {len(existing_facilities)}")
    print(f"APL candidates: {len(apl_candidates)}")
    print(f"Total candidate facilities: {len(facilities)}")

    print("Building road-network distance and cost matrices...")
    dist_matrix = build_road_distance_matrix(cbs_demand, facilities, nodes, edges)

    pickup_probability_scale = calibrate_pickup_scale(
        cbs_demand,
        activity,
        dist_matrix,
        len(existing_facilities),
    )

    print(f"Pickup probability scale: {pickup_probability_scale:.3f}")

    assignment_cost, daily_pickups = build_cost_matrices(
        cbs_demand,
        facilities,
        dist_matrix,
        pickup_probability_scale,
    )

    print("Solving MILP...")
    open_solution, assignment_solution, status, objective_value = solve_mip(
        cbs_demand,
        facilities,
        dist_matrix,
        assignment_cost,
        daily_pickups,
        protected_ids,
    )

    print(f"\nSolver status: {status}")
    print(f"Solver objective: EUR {objective_value:,.0f}")

    summary = evaluate_solution_components(
        cbs_demand,
        facilities,
        open_solution,
        assignment_solution,
        dist_matrix,
        pickup_probability_scale,
    )

    summary["solver_status"] = status
    summary["solver_objective"] = objective_value
    summary["apl_candidate_count"] = N_APL_CANDIDATES

    print("\nMILP solution summary")
    print(f"Total cost: EUR {summary['total_cost']:,.0f}")
    print(f"Fixed cost: EUR {summary['fixed_cost']:,.0f}")
    print(f"Delivery cost: EUR {summary['delivery_cost']:,.0f}")
    print(f"Storage cost: EUR {summary['storage_cost']:,.0f}")
    print(f"APL operating cost: EUR {summary['apl_operating_cost']:,.0f}")
    print(f"Penalty cost: EUR {summary['penalty_cost']:,.0f}")
    print(f"Open locations: {summary['open_locations']}")
    print(f"Open SPs: {summary['open_service_points']}")
    print(f"Open APLs: {summary['open_apls']}")
    print(f"Annual pickups: {summary['annual_pickups']:,.0f}")
    print(f"Annual deliveries: {summary['annual_deliveries']:,.0f}")
    print(f"Pickup share: {summary['pickup_share']:.2%}")
    print(f"Average distance: {summary['avg_distance_km']:.2f} km")
    print(f"Max distance: {summary['max_distance_km']:.2f} km")
    print(f"Population within 1 km: {summary['population_within_1km']:.2%}")
    print(f"Population within 2 km: {summary['population_within_2km']:.2%}")
    print(f"Population over 3 km: {summary['population_over_3km']:.2%}")
    print(f"City bounce rate: {summary['city_bounce_rate']:.3%}")
    print(f"Max location bounce rate: {summary['max_location_bounce_rate']:.3%}")

    save_outputs(
        cbs_demand,
        facilities,
        open_solution,
        assignment_solution,
        summary,
        sp,
        cbs,
    )

    facility_summary = summarize_facilities(facilities, open_solution)
    print_decisions(facility_summary)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

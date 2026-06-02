from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


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
OUTPUT_DIR = Path("/Users/macbook/Desktop/outputs")


RANDOM_SEED = 42

# Costs from the case / assumptions
FIXED_COST_SERVICE_POINT = 50_000          # euro per year
FIXED_COST_APL = 15_000                    # assumption: APL cheaper than full SP
DELIVERY_COST_PER_PACKAGE_KM = 1.50        # euro per package per km
STORAGE_COST_PER_PACKAGE_DAY = 0.10        # euro per package per day
AVERAGE_PICKUP_STORAGE_DAYS = 2.0
APL_SERVICE_COST_PER_PICKUP = 0.25
APL_REPLENISHMENT_COST_PER_PACKAGE_KM = 0.40

# assumption

# Capacity / bounce assumptions
DEFAULT_SP_CAPACITY_DAILY = 350            # packages/day, fallback if no observed capacity
APL_CAPACITY_DAILY = 90                    # assumption: APL has lower capacity
CITY_BOUNCE_LIMIT = 0.01                   # < 1%
LOCATION_BOUNCE_LIMIT = 0.02               # < 2%

# Penalties
CITY_BOUNCE_PENALTY = 2_000_000
LOCATION_BOUNCE_PENALTY = 5_000_000
NO_ASSIGNMENT_PENALTY = 10_000_000
LONG_DISTANCE_PENALTY_PER_PACKAGE_KM = 0.25

# GA parameters
POPULATION_SIZE = 180
GENERATIONS = 100
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.08
ELITE_SIZE = 4
LOCAL_SEARCH_PASSES = 1
RANDOM_IMMIGRANTS = 20
LOCAL_SEARCH_PROBABILITY = 0.25
STAGNATION_LIMIT = 10

# Network design assumptions
MIN_OPEN_LOCATIONS = 1
MAX_OPEN_LOCATIONS = 47
PROTECTED_TOP_SP_COUNT = 0
SP_CLOSURE_PENALTY = 10_000
APL_COMPLEXITY_PENALTY = 3_000
N_APL_CANDIDATES = 12
MIN_DISTANCE_BETWEEN_APL_CANDIDATES_KM = 0.8
PICKUP_PROBABILITY_SCALE = 1.0



@dataclass
class Facility:
    facility_id: str
    facility_type: str       # "SP" or "APL"
    x: float
    y: float
    fixed_cost: float
    capacity_daily: float


@dataclass
class EvaluationResult:
    total_cost: float
    fixed_cost: float
    delivery_cost: float
    storage_cost: float
    apl_operating_cost: float
    closure_penalty: float
    apl_complexity_penalty: float
    penalty_cost: float
    city_bounce_rate: float
    max_location_bounce_rate: float
    open_locations: int
    annual_pickups: float
    annual_deliveries: float
    avg_distance_km: float





def load_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sp = pd.read_excel(path, sheet_name="Service Point Locations")
    activity = pd.read_excel(path, sheet_name="Daily Activity")
    cbs = pd.read_excel(path, sheet_name="CBS Squares")
    nodes = pd.read_excel(path, sheet_name="Nodes")
    edges = pd.read_excel(path, sheet_name="Edges")
    return sp, activity, cbs, nodes, edges



def estimate_cbs_demand(cbs: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    cbs_known = cbs.dropna(subset=["Population", "X", "Y"]).copy()

    total_annual_demand = (activity["Deliveries"] + activity["Pickups"]).sum()
    total_population = cbs_known["Population"].sum()

    cbs_known["demand_weight"] = cbs_known["Population"] / total_population
    cbs_known["annual_demand"] = cbs_known["demand_weight"] * total_annual_demand

    return cbs_known


def build_existing_facilities(sp: pd.DataFrame, activity: pd.DataFrame) -> List[Facility]:

    activity_by_sp = (
        activity.groupby("Location ID", as_index=False)
        .agg(actual_pickups=("Pickups", "sum"), actual_deliveries=("Deliveries", "sum"))
    )
    activity_by_sp["avg_daily_pickups"] = activity_by_sp["actual_pickups"] / 365

    sp2 = sp.merge(activity_by_sp, on="Location ID", how="left")
    sp2["avg_daily_pickups"] = sp2["avg_daily_pickups"].fillna(0)

    facilities = []
    for _, row in sp2.iterrows():
        cap = max(DEFAULT_SP_CAPACITY_DAILY, 1.35 * row["avg_daily_pickups"])
        facilities.append(
            Facility(
                facility_id=f"SP_{int(row['Location ID'])}",
                facility_type="SP",
                x=float(row["X"]),
                y=float(row["Y"]),
                fixed_cost=FIXED_COST_SERVICE_POINT,
                capacity_daily=float(cap),
            )
        )
    return facilities

def get_protected_sp_ids(activity: pd.DataFrame) -> set[str]:

    busiest = (
        activity.groupby("Location ID", as_index=False)
        .agg(total_activity=("Deliveries", "sum"))
    )

    pickups = (
        activity.groupby("Location ID", as_index=False)
        .agg(total_pickups=("Pickups", "sum"))
    )

    busiest = busiest.merge(pickups, on="Location ID")
    busiest["total_activity"] = busiest["total_activity"] + busiest["total_pickups"]

    top_ids = (
        busiest.sort_values("total_activity", ascending=False)
        .head(PROTECTED_TOP_SP_COUNT)["Location ID"]
        .astype(int)
        .tolist()
    )

    return {f"SP_{loc_id}" for loc_id in top_ids}

def distance_km(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * 0.1 / 1000

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

    graph = csr_matrix(
        (weights, (row_idx, col_idx)),
        shape=(len(nodes), len(nodes))
    )

    node_xy = nodes[["X", "Y"]].to_numpy(dtype=float)

    return graph, node_id_to_idx, node_xy


def nearest_node_indices(points_x: np.ndarray, points_y: np.ndarray, node_xy: np.ndarray) -> np.ndarray:

    tree = cKDTree(node_xy)
    points = np.column_stack([points_x, points_y])
    _, nearest_indices = tree.query(points)
    return nearest_indices


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
        node_xy
    )

    facility_x = np.array([f.x for f in facilities])
    facility_y = np.array([f.y for f in facilities])

    facility_node_idx = nearest_node_indices(
        facility_x,
        facility_y,
        node_xy
    )

    unique_cbs_nodes, inverse_cbs = np.unique(cbs_node_idx, return_inverse=True)

    shortest_paths = dijkstra(
        csgraph=graph,
        directed=True,
        indices=unique_cbs_nodes
    )

    road_dist_matrix = shortest_paths[inverse_cbs][:, facility_node_idx]

    fallback_dist_matrix = build_straight_line_distance_matrix(cbs_demand, facilities)

    road_dist_matrix = np.where(
        np.isfinite(road_dist_matrix),
        road_dist_matrix,
        fallback_dist_matrix
    )

    return road_dist_matrix


def generate_apl_candidates(cbs_demand: pd.DataFrame, existing_facilities: List[Facility]) -> List[Facility]:

    cbs_x = cbs_demand["X"].to_numpy()
    cbs_y = cbs_demand["Y"].to_numpy()

    sp_x = np.array([f.x for f in existing_facilities])
    sp_y = np.array([f.y for f in existing_facilities])

    dist_matrix = distance_km(cbs_x[:, None], cbs_y[:, None], sp_x[None, :], sp_y[None, :])
    nearest_dist = dist_matrix.min(axis=1)

    candidates = cbs_demand.copy()
    candidates["nearest_sp_dist_km"] = nearest_dist
    candidates["apl_score"] = candidates["Population"] * candidates["nearest_sp_dist_km"]
    candidates = candidates.sort_values("apl_score", ascending=False).reset_index(drop=True)

    selected_rows = []

    for _, row in candidates.iterrows():
        if len(selected_rows) >= N_APL_CANDIDATES:
            break

        if not selected_rows:
            selected_rows.append(row)
            continue

        selected_x = np.array([r["X"] for r in selected_rows])
        selected_y = np.array([r["Y"] for r in selected_rows])

        dist_to_selected = distance_km(
            np.array([row["X"]]),
            np.array([row["Y"]]),
            selected_x,
            selected_y,
        )

        if dist_to_selected.min() >= MIN_DISTANCE_BETWEEN_APL_CANDIDATES_KM:
            selected_rows.append(row)

    if len(selected_rows) < N_APL_CANDIDATES:
        selected_squares = {r["Square"] for r in selected_rows}
        for _, row in candidates.iterrows():
            if len(selected_rows) >= N_APL_CANDIDATES:
                break
            if row["Square"] not in selected_squares:
                selected_rows.append(row)
                selected_squares.add(row["Square"])

    apls = []
    for idx, row in enumerate(selected_rows):
        apls.append(
            Facility(
                facility_id=f"APL_{idx + 1}_{row['Square']}",
                facility_type="APL",
                x=float(row["X"]),
                y=float(row["Y"]),
                fixed_cost=FIXED_COST_APL,
                capacity_daily=APL_CAPACITY_DAILY,
            )
        )

    return apls


def base_pickup_probability(dist_km: np.ndarray) -> np.ndarray:
    return np.select(
        [
            dist_km <= 0.5,
            dist_km <= 1.0,
            dist_km <= 2.0,
        ],
        [
            0.80,
            0.65,
            0.45,
        ],
        default=0.25,
    )


def pickup_probability(dist_km: np.ndarray) -> np.ndarray:
    return np.clip(
        base_pickup_probability(dist_km) * PICKUP_PROBABILITY_SCALE,
        0.05,
        0.95,
    )


def calibrate_pickup_probability_scale(
    cbs_demand: pd.DataFrame,
    activity: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
) -> float:
    demand = cbs_demand["annual_demand"].to_numpy()

    baseline_chromosome = np.array(
        [1 if f.facility_type == "SP" else 0 for f in facilities],
        dtype=int,
    )

    open_idx = np.where(baseline_chromosome == 1)[0]
    open_distances = dist_matrix[:, open_idx]
    assigned_dist = open_distances.min(axis=1)

    raw_prob = base_pickup_probability(assigned_dist)
    raw_pickup_share = np.average(raw_prob, weights=demand)

    observed_pickup_share = activity["Pickups"].sum() / (
        activity["Pickups"].sum() + activity["Deliveries"].sum()
    )

    if raw_pickup_share <= 0:
        return 1.0

    return float(observed_pickup_share / raw_pickup_share)


def build_straight_line_distance_matrix(cbs_demand: pd.DataFrame, facilities: List[Facility]) -> np.ndarray:
    cbs_x = cbs_demand["X"].to_numpy()
    cbs_y = cbs_demand["Y"].to_numpy()
    fac_x = np.array([f.x for f in facilities])
    fac_y = np.array([f.y for f in facilities])
    return distance_km(cbs_x[:, None], cbs_y[:, None], fac_x[None, :], fac_y[None, :])


def repair_chromosome(
    chromosome: np.ndarray,
    facilities: List[Facility],
    protected_sp_ids: set[str],
) -> np.ndarray:
    chromosome = chromosome.copy()

    protected_indices = np.array(
        [
            i for i, f in enumerate(facilities)
            if f.facility_id in protected_sp_ids
        ],
        dtype=int
    )

    if len(protected_indices) > 0:
        chromosome[protected_indices] = 1

    if chromosome.sum() < MIN_OPEN_LOCATIONS:
        closed = np.where(chromosome == 0)[0]
        if len(closed) > 0:
            chromosome[np.random.choice(closed)] = 1

    while chromosome.sum() > MAX_OPEN_LOCATIONS:
        closable = np.array(
            [
                i for i, f in enumerate(facilities)
                if chromosome[i] == 1 and f.facility_id not in protected_sp_ids
            ],
            dtype=int
        )

        if len(closable) == 0:
            break

        chromosome[np.random.choice(closable)] = 0

    if len(protected_indices) > 0:
        chromosome[protected_indices] = 1

    return chromosome


def evaluate_solution(
    chromosome: np.ndarray,
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
) -> EvaluationResult:
    open_idx = np.where(chromosome == 1)[0]

    if len(open_idx) == 0:
        return EvaluationResult(
            total_cost=NO_ASSIGNMENT_PENALTY,
            fixed_cost=0,
            delivery_cost=0,
            storage_cost=0,
            apl_operating_cost=0,
            closure_penalty=0,
            apl_complexity_penalty=0,
            penalty_cost=NO_ASSIGNMENT_PENALTY,
            city_bounce_rate=1,
            max_location_bounce_rate=1,
            open_locations=0,
            annual_pickups=0,
            annual_deliveries=0,
            avg_distance_km=999,
        )

    demand = cbs_demand["annual_demand"].to_numpy()

    # Assign each CBS square to nearest open facility.
    open_distances = dist_matrix[:, open_idx]
    nearest_open_position = open_distances.argmin(axis=1)
    assigned_facility_idx = open_idx[nearest_open_position]
    assigned_dist = open_distances[np.arange(open_distances.shape[0]), nearest_open_position]

    p_pick = pickup_probability(assigned_dist)
    annual_pickups = demand * p_pick
    annual_deliveries = demand * (1 - p_pick)

    # Costs
    fixed_cost = sum(facilities[j].fixed_cost for j in open_idx)
    delivery_cost = np.sum(annual_deliveries * assigned_dist * DELIVERY_COST_PER_PACKAGE_KM)
    storage_cost = np.sum(annual_pickups * STORAGE_COST_PER_PACKAGE_DAY * AVERAGE_PICKUP_STORAGE_DAYS)
    facility_types = np.array([f.facility_type for f in facilities])
    assigned_is_apl = facility_types[assigned_facility_idx] == "APL"
    apl_service_cost = np.sum(annual_pickups[assigned_is_apl] * APL_SERVICE_COST_PER_PICKUP)
    apl_replenishment_cost = np.sum(
        annual_pickups[assigned_is_apl] * assigned_dist[assigned_is_apl] * APL_REPLENISHMENT_COST_PER_PACKAGE_KM
    )
    apl_operating_cost = apl_service_cost + apl_replenishment_cost
    open_mask = chromosome == 1
    facility_types = np.array([f.facility_type for f in facilities])

    closed_existing_sp_count = np.sum((facility_types == "SP") & (~open_mask))
    open_apl_count = np.sum((facility_types == "APL") & open_mask)

    closure_penalty = closed_existing_sp_count * SP_CLOSURE_PENALTY
    apl_complexity_penalty = open_apl_count * APL_COMPLEXITY_PENALTY

    # Optional penalty for very long access distances.
    long_distance_extra = np.maximum(assigned_dist - 2.0, 0)
    long_distance_penalty = np.sum(demand * long_distance_extra * LONG_DISTANCE_PENALTY_PER_PACKAGE_KM)

    # Capacity and bounce
    pickup_by_facility_annual = np.zeros(len(facilities))
    np.add.at(pickup_by_facility_annual, assigned_facility_idx, annual_pickups)
    pickup_by_facility_daily = pickup_by_facility_annual / 365

    capacities = np.array([f.capacity_daily for f in facilities])
    overflow_daily = np.maximum(pickup_by_facility_daily - capacities, 0)

    total_pickups_daily = pickup_by_facility_daily.sum()
    total_overflow_daily = overflow_daily.sum()

    city_bounce_rate = total_overflow_daily / total_pickups_daily if total_pickups_daily > 0 else 0

    location_bounce_rates = np.divide(
        overflow_daily,
        pickup_by_facility_daily,
        out=np.zeros_like(overflow_daily),
        where=pickup_by_facility_daily > 0,
    )
    max_location_bounce_rate = location_bounce_rates.max()

    city_bounce_penalty = max(city_bounce_rate - CITY_BOUNCE_LIMIT, 0) * CITY_BOUNCE_PENALTY
    location_bounce_penalty = np.sum(
        np.maximum(location_bounce_rates - LOCATION_BOUNCE_LIMIT, 0)
    ) * LOCATION_BOUNCE_PENALTY

    penalty_cost = long_distance_penalty + city_bounce_penalty + location_bounce_penalty
    total_cost = (
            fixed_cost
            + delivery_cost
            + storage_cost
            + apl_operating_cost
            + closure_penalty
            + apl_complexity_penalty
            + penalty_cost
    )

    avg_distance_km = np.average(assigned_dist, weights=demand)

    return EvaluationResult(
        total_cost=float(total_cost),
        fixed_cost=float(fixed_cost),
        delivery_cost=float(delivery_cost),
        storage_cost=float(storage_cost),
        apl_operating_cost=float(apl_operating_cost),
        closure_penalty=float(closure_penalty),
        apl_complexity_penalty=float(apl_complexity_penalty),
        penalty_cost=float(penalty_cost),
        city_bounce_rate=float(city_bounce_rate),
        max_location_bounce_rate=float(max_location_bounce_rate),
        open_locations=int(chromosome.sum()),
        annual_pickups=float(annual_pickups.sum()),
        annual_deliveries=float(annual_deliveries.sum()),
        avg_distance_km=float(avg_distance_km),
    )


def random_chromosome(
    n_facilities: int,
    facilities: List[Facility],
    protected_sp_ids: set[str],
) -> np.ndarray:
    chrom = np.zeros(n_facilities, dtype=int)
    n_open = random.randint(MIN_OPEN_LOCATIONS, min(MAX_OPEN_LOCATIONS, n_facilities))
    open_idx = np.random.choice(np.arange(n_facilities), size=n_open, replace=False)
    chrom[open_idx] = 1
    return repair_chromosome(chrom, facilities, protected_sp_ids)

def shake_chromosome(
    chromosome: np.ndarray,
    facilities: List[Facility],
    protected_sp_ids: set[str],
    strength: int = 8,
) -> np.ndarray:
    shaken = chromosome.copy()

    closable = np.array(
        [
            i for i, f in enumerate(facilities)
            if shaken[i] == 1 and f.facility_id not in protected_sp_ids
        ],
        dtype=int,
    )
    closed = np.where(shaken == 0)[0]

    n_swap = min(strength, len(closable), len(closed))

    if n_swap > 0:
        close_idx = np.random.choice(closable, size=n_swap, replace=False)
        open_idx = np.random.choice(closed, size=n_swap, replace=False)

        shaken[close_idx] = 0
        shaken[open_idx] = 1

    return repair_chromosome(shaken, facilities, protected_sp_ids)



def tournament_selection(population: List[np.ndarray], costs: List[float]) -> np.ndarray:
    idx = np.random.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    best_idx = min(idx, key=lambda i: costs[i])
    return population[best_idx].copy()


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    facilities: List[Facility],
    protected_sp_ids: set[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    mask = np.random.rand(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return (
        repair_chromosome(child1, facilities, protected_sp_ids),
        repair_chromosome(child2, facilities, protected_sp_ids),
    )


def mutate(
    chromosome: np.ndarray,
    facilities: List[Facility],
    protected_sp_ids: set[str],
) -> np.ndarray:
    mutated = chromosome.copy()

    flip_mask = np.random.rand(len(mutated)) < MUTATION_RATE
    mutated[flip_mask] = 1 - mutated[flip_mask]

    if random.random() < 0.35:
        open_idx = np.array(
            [
                i for i, f in enumerate(facilities)
                if mutated[i] == 1 and f.facility_id not in protected_sp_ids
            ],
            dtype=int,
        )
        closed_idx = np.where(mutated == 0)[0]

        if len(open_idx) > 0 and len(closed_idx) > 0:
            close_i = np.random.choice(open_idx)
            open_i = np.random.choice(closed_idx)
            mutated[close_i] = 0
            mutated[open_i] = 1

    return repair_chromosome(mutated, facilities, protected_sp_ids)



def local_search(
    chromosome: np.ndarray,
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
    protected_sp_ids: set[str],
) -> np.ndarray:

    best = chromosome.copy()
    best_eval = evaluate_solution(best, cbs_demand, facilities, dist_matrix)

    for _ in range(LOCAL_SEARCH_PASSES):
        improved = False
        candidate_indices = np.random.permutation(len(facilities))

        for idx in candidate_indices:
            facility = facilities[idx]

            if best[idx] == 1 and facility.facility_id in protected_sp_ids:
                continue

            trial = best.copy()
            trial[idx] = 1 - trial[idx]
            trial = repair_chromosome(trial, facilities, protected_sp_ids)

            trial_eval = evaluate_solution(trial, cbs_demand, facilities, dist_matrix)

            if trial_eval.total_cost < best_eval.total_cost:
                best = trial
                best_eval = trial_eval
                improved = True

        if not improved:
            break

    return best

def swap_local_search(
    chromosome: np.ndarray,
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
    protected_sp_ids: set[str],
) -> np.ndarray:

    best = chromosome.copy()
    best_eval = evaluate_solution(best, cbs_demand, facilities, dist_matrix)

    open_idx = [
        i for i, f in enumerate(facilities)
        if best[i] == 1 and f.facility_id not in protected_sp_ids
    ]
    closed_idx = [
        i for i in range(len(facilities))
        if best[i] == 0
    ]

    random.shuffle(open_idx)
    random.shuffle(closed_idx)

    for close_i in open_idx:
        for open_i in closed_idx:
            trial = best.copy()
            trial[close_i] = 0
            trial[open_i] = 1
            trial = repair_chromosome(trial, facilities, protected_sp_ids)

            trial_eval = evaluate_solution(trial, cbs_demand, facilities, dist_matrix)

            if trial_eval.total_cost < best_eval.total_cost:
                best = trial
                best_eval = trial_eval

    return best

def run_ga(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    dist_matrix: np.ndarray,
    protected_sp_ids: set[str],
) -> Tuple[np.ndarray, EvaluationResult, pd.DataFrame]:
    n_facilities = len(facilities)

    population = [
        random_chromosome(n_facilities, facilities, protected_sp_ids)
        for _ in range(POPULATION_SIZE)
    ]

    history = []
    best_chromosome = None
    best_eval = None
    generations_without_improvement = 0

    for generation in range(GENERATIONS):
        evaluations = [
            evaluate_solution(chrom, cbs_demand, facilities, dist_matrix)
            for chrom in population
        ]
        costs = [ev.total_cost for ev in evaluations]

        generation_best_idx = int(np.argmin(costs))
        generation_best = evaluations[generation_best_idx]

        if best_eval is None or generation_best.total_cost < best_eval.total_cost:
            best_eval = generation_best
            best_chromosome = population[generation_best_idx].copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        history.append(
            {
                "generation": generation,
                "best_cost": generation_best.total_cost,
                "mean_cost": float(np.mean(costs)),
                "overall_best_cost": best_eval.total_cost,
                "open_locations": generation_best.open_locations,
                "avg_distance_km": generation_best.avg_distance_km,
                "city_bounce_rate": generation_best.city_bounce_rate,
            }
        )

        # Elitism
        elite_indices = np.argsort(costs)[:ELITE_SIZE]
        new_population = [population[i].copy() for i in elite_indices]

        # Create children
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, costs)
            p2 = tournament_selection(population, costs)
            c1, c2 = crossover(p1, p2, facilities, protected_sp_ids)
            c1 = mutate(c1, facilities, protected_sp_ids)
            c2 = mutate(c2, facilities, protected_sp_ids)

            if random.random() < LOCAL_SEARCH_PROBABILITY:
                c1 = local_search(c1, cbs_demand, facilities, dist_matrix, protected_sp_ids)
                c1 = swap_local_search(c1, cbs_demand, facilities, dist_matrix, protected_sp_ids)

            if random.random() < LOCAL_SEARCH_PROBABILITY:
                c2 = local_search(c2, cbs_demand, facilities, dist_matrix, protected_sp_ids)
                c2 = swap_local_search(c2, cbs_demand, facilities, dist_matrix, protected_sp_ids)

            new_population.append(c1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(c2)
        for _ in range(RANDOM_IMMIGRANTS):
            replace_idx = random.randrange(ELITE_SIZE, len(new_population))
            new_population[replace_idx] = random_chromosome(
                n_facilities,
                facilities,
                protected_sp_ids,
            )

        if generations_without_improvement >= STAGNATION_LIMIT:
            print(f"Stagnation at generation {generation}; shaking population...")
            for _ in range(POPULATION_SIZE // 2):
                replace_idx = random.randrange(ELITE_SIZE, len(new_population))

                if best_chromosome is not None and random.random() < 0.6:
                    new_population[replace_idx] = shake_chromosome(
                        best_chromosome,
                        facilities,
                        protected_sp_ids,
                        strength=random.randint(6, 14),
                    )
                else:
                    new_population[replace_idx] = random_chromosome(
                        n_facilities,
                        facilities,
                        protected_sp_ids,
                    )

            generations_without_improvement = 0

        population = new_population

        if generation % 10 == 0 or generation == GENERATIONS - 1:
            print(
                f"Generation {generation:03d} | "
                f"best cost: €{generation_best.total_cost:,.0f} | "
                f"overall best: €{best_eval.total_cost:,.0f} | "
                f"open: {generation_best.open_locations}"
            )

    return best_chromosome, best_eval, pd.DataFrame(history)


def summarize_solution(
    chromosome: np.ndarray,
    facilities: List[Facility],
    result: EvaluationResult,
) -> pd.DataFrame:
    rows = []
    for gene, facility in zip(chromosome, facilities):
        rows.append(
            {
                "facility_id": facility.facility_id,
                "facility_type": facility.facility_type,
                "open": int(gene),
                "x": facility.x,
                "y": facility.y,
                "fixed_cost": facility.fixed_cost,
                "capacity_daily": facility.capacity_daily,
            }
        )
    return pd.DataFrame(rows)


def plot_convergence(history: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping convergence plot.")
        return

    plt.figure(figsize=(9, 5))
    plt.plot(history["generation"], history["overall_best_cost"], label="Best cost", color="#D7191C", linewidth=2.5)
    plt.plot(history["generation"], history["mean_cost"], label="Mean cost", color="#2F6F9F", alpha=0.55)
    plt.title("Genetic Algorithm Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Cost (€)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ga_convergence.png", dpi=300)
    plt.close()


def plot_selected_locations(
    cbs_demand: pd.DataFrame,
    facilities: List[Facility],
    chromosome: np.ndarray,
    output_dir: Path,
) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping network plot.")
        return

    facilities_df = summarize_solution(chromosome, facilities, None)

    plt.figure(figsize=(9, 7))
    plt.scatter(
        cbs_demand["X"],
        cbs_demand["Y"],
        s=np.clip(cbs_demand["Population"] / 12, 8, 160),
        color="#A0A7B0",
        alpha=0.35,
        label="CBS demand squares",
    )

    open_sp = facilities_df[(facilities_df["open"] == 1) & (facilities_df["facility_type"] == "SP")]
    open_apl = facilities_df[(facilities_df["open"] == 1) & (facilities_df["facility_type"] == "APL")]
    closed = facilities_df[facilities_df["open"] == 0]

    plt.scatter(closed["x"], closed["y"], s=30, color="#D1D5DB", alpha=0.45, label="Closed / inactive")
    plt.scatter(open_sp["x"], open_sp["y"], s=75, color="#D7191C", edgecolor="white", linewidth=0.8, label="Open service points")
    plt.scatter(open_apl["x"], open_apl["y"], s=95, color="#2F6F9F", edgecolor="white", linewidth=0.8, marker="^", label="Open APLs")

    plt.title("Best GA Network Configuration")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis("equal")
    plt.grid(alpha=0.18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ga_best_network.png", dpi=300)
    plt.close()

def convert_xy_to_rd(cbs: pd.DataFrame, x_values: pd.Series, y_values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:

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

def plot_current_vs_ga_real_map(
    sp: pd.DataFrame,
    cbs: pd.DataFrame,
    facilities: List[Facility],
    chromosome: np.ndarray,
    output_dir: Path,
) -> None:
    if plt is None or gpd is None or cx is None or Point is None:
        print("geopandas/contextily/pyproj/matplotlib not installed; skipping real map plot.")
        return

    solution_df = summarize_solution(chromosome, facilities, None)

    sp_map = sp.copy()
    sp_map["rd_x"], sp_map["rd_y"] = convert_xy_to_rd(cbs, sp_map["X"], sp_map["Y"])

    ga_map = solution_df.copy()
    ga_map["rd_x"], ga_map["rd_y"] = convert_xy_to_rd(cbs, ga_map["x"], ga_map["y"])

    sp_gdf = gpd.GeoDataFrame(
        sp_map,
        geometry=[Point(x, y) for x, y in zip(sp_map["rd_x"], sp_map["rd_y"])],
        crs="EPSG:28992",
    ).to_crs(epsg=3857)

    ga_gdf = gpd.GeoDataFrame(
        ga_map,
        geometry=[Point(x, y) for x, y in zip(ga_map["rd_x"], ga_map["rd_y"])],
        crs="EPSG:28992",
    ).to_crs(epsg=3857)

    kept_sp = ga_gdf[(ga_gdf["facility_type"] == "SP") & (ga_gdf["open"] == 1)]
    closed_sp = ga_gdf[(ga_gdf["facility_type"] == "SP") & (ga_gdf["open"] == 0)]
    opened_apl = ga_gdf[(ga_gdf["facility_type"] == "APL") & (ga_gdf["open"] == 1)]

    minx, miny, maxx, maxy = ga_gdf.total_bounds
    x_buffer = 2500
    y_buffer = 2500

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Current network
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

    closed_sp.plot(
        ax=axes[1],
        color="#737373",
        markersize=45,
        marker="x",
        linewidth=1.3,
        alpha=0.85,
        label="Closed existing SP",
    )

    kept_sp.plot(
        ax=axes[1],
        color="#D7191C",
        markersize=55,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.95,
        label="Kept existing SP",
    )

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

    n_kept_sp = len(kept_sp)
    n_opened_apl = len(opened_apl)

    axes[1].set_title(
        f"GA Optimized Network: {n_kept_sp} SPs + {n_opened_apl} APLs",
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
        except Exception as e:
            print(f"Could not load basemap: {e}")

        ax.set_axis_off()
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="#D9E2EC",
            fontsize=10,
        )

    fig.suptitle(
        "Current Network vs Genetic Algorithm Solution",
        fontsize=20,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.035,
        "The GA keeps selected existing service points, closes others, and opens APLs to reduce cost while maintaining accessibility.",
        ha="center",
        fontsize=11,
        color="#4B5C70",
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08, wspace=0.04)
    fig.savefig(output_dir / "ga_current_vs_optimized_real_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    best_chromosome: np.ndarray,
    best_eval: EvaluationResult,
    history: pd.DataFrame,
    facilities: List[Facility],
    cbs_demand: pd.DataFrame,
    sp: pd.DataFrame,
    cbs: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history.to_csv(OUTPUT_DIR / "ga_history.csv", index=False)

    solution_df = summarize_solution(best_chromosome, facilities, best_eval)
    solution_df.to_csv(OUTPUT_DIR / "ga_best_solution_facilities.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "total_cost": best_eval.total_cost,
                "fixed_cost": best_eval.fixed_cost,
                "delivery_cost": best_eval.delivery_cost,
                "storage_cost": best_eval.storage_cost,
                "penalty_cost": best_eval.penalty_cost,
                "city_bounce_rate": best_eval.city_bounce_rate,
                "max_location_bounce_rate": best_eval.max_location_bounce_rate,
                "open_locations": best_eval.open_locations,
                "annual_pickups": best_eval.annual_pickups,
                "annual_deliveries": best_eval.annual_deliveries,
                "avg_distance_km": best_eval.avg_distance_km,
                "apl_operating_cost": best_eval.apl_operating_cost,
                "closure_penalty": best_eval.closure_penalty,
                "apl_complexity_penalty": best_eval.apl_complexity_penalty,
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "ga_best_solution_summary.csv", index=False)

    plot_convergence(history, OUTPUT_DIR)
    plot_selected_locations(cbs_demand, facilities, best_chromosome, OUTPUT_DIR)
    plot_current_vs_ga_real_map(sp, cbs, facilities, best_chromosome, OUTPUT_DIR)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading data...")
    sp, activity, cbs, nodes, edges = load_data(DATA_PATH)

    print("Estimating CBS demand...")
    cbs_demand = estimate_cbs_demand(cbs, activity)

    print("Building facilities...")
    existing_facilities = build_existing_facilities(sp, activity)
    apl_candidates = generate_apl_candidates(cbs_demand, existing_facilities)
    facilities = existing_facilities + apl_candidates
    protected_sp_ids = get_protected_sp_ids(activity)
    print(f"Protected high-volume SPs: {sorted(protected_sp_ids)}")

    print(f"Existing service points: {len(existing_facilities)}")
    print(f"APL candidates: {len(apl_candidates)}")
    print(f"Total candidate facilities: {len(facilities)}")

    print("Building road-network distance matrix...")
    dist_matrix = build_road_distance_matrix(cbs_demand, facilities, nodes, edges)

    global PICKUP_PROBABILITY_SCALE
    PICKUP_PROBABILITY_SCALE = calibrate_pickup_probability_scale(
        cbs_demand,
        activity,
        facilities,
        dist_matrix,
    )

    print(f"Pickup probability scale: {PICKUP_PROBABILITY_SCALE:.3f}")

    print("Evaluating current network baseline...")
    baseline_chromosome = np.array(
        [1 if f.facility_type == "SP" else 0 for f in facilities],
        dtype=int,
    )
    baseline_eval = evaluate_solution(baseline_chromosome, cbs_demand, facilities, dist_matrix)
    print(f"Baseline cost: €{baseline_eval.total_cost:,.0f}")
    print(f"Baseline avg distance: {baseline_eval.avg_distance_km:.2f} km")
    print(f"Baseline open locations: {baseline_eval.open_locations}")

    print("\nRunning genetic algorithm...")
    best_chromosome, best_eval, history = run_ga(
        cbs_demand,
        facilities,
        dist_matrix,
        protected_sp_ids,
    )

    improvement = (baseline_eval.total_cost - best_eval.total_cost) / baseline_eval.total_cost

    print("\nBest GA solution")
    print(f"Total cost: €{best_eval.total_cost:,.0f}")
    print(f"Fixed cost: €{best_eval.fixed_cost:,.0f}")
    print(f"Delivery cost: €{best_eval.delivery_cost:,.0f}")
    print(f"Storage cost: €{best_eval.storage_cost:,.0f}")
    print(f"APL operating cost: €{best_eval.apl_operating_cost:,.0f}")
    print(f"Closure penalty: €{best_eval.closure_penalty:,.0f}")
    print(f"APL complexity penalty: €{best_eval.apl_complexity_penalty:,.0f}")
    print(f"Penalty cost: €{best_eval.penalty_cost:,.0f}")
    print(f"Open locations: {best_eval.open_locations}")
    print(f"Annual pickups: {best_eval.annual_pickups:,.0f}")
    print(f"Annual deliveries: {best_eval.annual_deliveries:,.0f}")
    print(f"Average distance: {best_eval.avg_distance_km:.2f} km")
    print(f"City bounce rate: {best_eval.city_bounce_rate:.3%}")
    print(f"Max location bounce rate: {best_eval.max_location_bounce_rate:.3%}")
    print(f"Improvement vs baseline: {improvement:.2%}")

    print("\nSaving outputs...")
    save_outputs(best_chromosome, best_eval, history, facilities, cbs_demand, sp, cbs)
    print(f"Saved to: {OUTPUT_DIR}")

    solution_df = summarize_solution(best_chromosome, facilities, best_eval)

    open_sp = solution_df[
        (solution_df["facility_type"] == "SP") & (solution_df["open"] == 1)
    ]["facility_id"].str.replace("SP_", "").tolist()

    closed_sp = solution_df[
        (solution_df["facility_type"] == "SP") & (solution_df["open"] == 0)
    ]["facility_id"].str.replace("SP_", "").tolist()

    open_apl = solution_df[
        (solution_df["facility_type"] == "APL") & (solution_df["open"] == 1)
    ]["facility_id"].tolist()

    closed_apl = solution_df[
        (solution_df["facility_type"] == "APL") & (solution_df["open"] == 0)
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

if __name__ == "__main__":
     main()


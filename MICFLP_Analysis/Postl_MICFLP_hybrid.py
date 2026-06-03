"""
Post&L Maastricht — Hybrid SP + APL Facility Location Model
============================================================

WHAT THIS MODEL DOES
---------------------
Extends the SP-only MCFLP (v2) to include Automated Parcel Lockers (APLs)
as a second, cheaper facility type. The model jointly decides:

  1. Which of the 35 existing SPs to keep open
  2. Which CBS squares to place a new APL in
  3. Which facility (SP or APL) each CBS demand zone is assigned to

This directly answers the three case questions:
  - How much money is being lost without APLs?
  - What is the optimal hybrid network layout?
  - How should APLs be incorporated and where?

HOW SPs AND APLs DIFFER
------------------------
                    SP              APL
  Fixed cost/yr   €50,000         €15,000   (assumption: cheaper hardware, no staff)
  Capacity        500 pkg peak    120 pkg peak  (smaller locker bank)
  Pickup boost    —               +15% on base p_pickup  (24/7 access)
  Delivery?       Yes             No  (APL is pickup-only: packages still
                                       delivered to nearest SP if no APL)
  Candidates      35 existing     186 CBS squares without an SP

APL PICKUP PROBABILITY
----------------------
Because APLs are open 24/7, customers who would otherwise miss SP opening
hours are more likely to pick up. We model this as a multiplicative boost:

  p_pickup_APL(d) = min(1.0,  p_pickup_SP(d) × (1 + APL_PICKUP_BOOST))
                 = min(1.0,  [1/(1 + 0.639×d)] × 1.15)

Calibrated assumption: APLs increase pickup share by 15% relative to an SP
at the same distance. This is conservative — real-world data suggests
20-25% for 24/7 lockers in urban areas, but 15% is defensible for the case.

APL DEMOGRAPHIC AFFINITY (optional scoring, not a hard constraint)
------------------------------------------------------------------
CBS squares with higher APL affinity are prioritised as candidates.
Score = 0.4 × young_adult_share (Age15-44)
      + 0.3 × urbanisation_index (1-5, normalised)
      + 0.3 × income_score (higher income → more online shopping)

Squares scoring > median affinity are included as APL candidates.
Set APL_USE_AFFINITY_FILTER = False to allow all 186 squares.

COST PARAMETERS
---------------
  Fixed cost SP           : €50,000 / yr
  Fixed cost APL          : €15,000 / yr
  Delivery cost           : €1.50 / km / parcel (SP catchment only)
  Storage cost            : €0.10 / pkg / day × 1.7 days = €0.17 / pickup
  Pickup savings          : €1.01 / pickup (avoided delivery)
  Net pickup benefit      : €0.84 / pickup

APL candidate locations: 186 CBS squares without an existing SP.
SP candidates: 35 existing SP locations.
"""

import warnings
import numpy as np
import pandas as pd
import heapq
from scipy.spatial import KDTree
import pulp

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH        = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/data_Maastricht_2025.xlsx"
DIST_MATRIX_PATH = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/outputs-analysis/network_dist_matrix.csv"
OUTPUT_DIR       = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/outputs-analysis/"

# ── Cost parameters ──────────────────────────────────────────────────────────
FIXED_COST_SP        = 50_000   # € / yr
FIXED_COST_APL       = 15_000   # € / yr  ← KEY ASSUMPTION (tune this)
DELIVERY_COST_KM     = 1.50     # € / km / delivery
STORAGE_COST_PKG_DAY = 0.10     # € / pkg / day
AVG_STORAGE_DAYS     = 1.7
DELIVERY_SAVES       = 1.01     # € saved per pickup

# ── Pickup probability parameters ────────────────────────────────────────────
PICKUP_DECAY         = 0.639    # calibrated: p=0.643 at 0.87 km (EDA avg)
APL_PICKUP_BOOST     = 0.15     # APLs boost pickup probability by 15% (24/7 access)

# ── Capacity parameters ───────────────────────────────────────────────────────
PEAK_RATIO           = 2.34     # EDA: avg peak/avg daily pickup ratio
SP_STORAGE_CAPACITY  = 500      # max packages at SP on peak day
APL_STORAGE_CAPACITY = 120      # max packages at APL on peak day (smaller unit)

# ── APL placement options ─────────────────────────────────────────────────────
APL_USE_AFFINITY_FILTER = True  # True = only demographically suitable squares
APL_MAX_OPEN         = None     # max number of APLs to open (None = unlimited)
SP_MIN_OPEN          = None     # min SPs that must stay open
SP_MAX_OPEN          = None     # max SPs that can stay open

# ── Bounce rate thresholds (post-solve diagnostics) ───────────────────────────
SP_BOUNCE_THRESHOLD   = 0.02
CITY_BOUNCE_THRESHOLD = 0.01

SOLVER_TIME_LIMIT = 180

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
xl  = pd.ExcelFile(DATA_PATH)

sp_df = pd.read_excel(xl, "Service Point Locations").rename(
          columns={"Location ID": "sp_id", "X": "x", "Y": "y",
                   "Containing Square": "square"})

cbs   = pd.read_excel(xl, "CBS Squares").rename(columns={"Square": "square"})
cbs   = cbs.dropna(subset=["Population", "X", "Y"])
cbs   = cbs[cbs["Population"] > 0].reset_index(drop=True)
cbs   = cbs.rename(columns={"X": "x", "Y": "y"})

act   = pd.read_excel(xl, "Daily Activity").rename(columns={"Location ID": "sp_id"})

TOTAL_ANNUAL_PARCELS = act["Deliveries"].sum() + act["Pickups"].sum()
print(f"  Existing SPs          : {len(sp_df)}")
print(f"  CBS demand zones      : {len(cbs)}")
print(f"  Total annual parcels  : {TOTAL_ANNUAL_PARCELS:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD APL CANDIDATE LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding APL candidate locations …")

sp_squares   = set(sp_df["square"].tolist())
apl_raw      = cbs[~cbs["square"].isin(sp_squares)].copy().reset_index(drop=True)

# ── Demographic affinity scoring ─────────────────────────────────────────────
def income_score(val):
    """Map Dutch income category strings to 0–1 score."""
    mapping = {
        "00-20 low": 0.1,
        "00-40 laag tot onder midden": 0.2,
        "20-40 below middle": 0.3,
        "20-60 below middle to middle": 0.4,
        "40-60 midden": 0.5,
        "40-80 middle to above middle": 0.65,
        "60-80 above middle": 0.75,
        "60-100 above middle to high": 0.85,
        "80-100 high": 1.0,
    }
    return mapping.get(str(val).strip(), 0.5)

apl_raw["income_sc"] = apl_raw["Median household income"].apply(income_score)

# Young adults share (Age 15-44) — higher = more APL-likely (tech-comfortable, busy)
apl_raw["young_adult_share"] = (
    apl_raw[["Age15-24", "Age25-44"]].sum(axis=1).fillna(0) /
    apl_raw["Population"].replace(0, np.nan)
).fillna(0)

# Urbanisation index 1-5 (normalised to 0-1)
apl_raw["urban_norm"] = (apl_raw["Urbanization index"].fillna(3) - 1) / 4.0

# Composite affinity score
apl_raw["apl_affinity"] = (
    0.40 * apl_raw["young_adult_share"] +
    0.30 * apl_raw["urban_norm"] +
    0.30 * apl_raw["income_sc"]
)

if APL_USE_AFFINITY_FILTER:
    threshold = apl_raw["apl_affinity"].median()
    apl_cands = apl_raw[apl_raw["apl_affinity"] >= threshold].copy().reset_index(drop=True)
    print(f"  APL candidates (affinity ≥ median {threshold:.3f}): {len(apl_cands)}")
else:
    apl_cands = apl_raw.copy().reset_index(drop=True)
    print(f"  APL candidates (all non-SP squares): {len(apl_cands)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DIJKSTRA — compute distances from APL candidate nodes
# ═══════════════════════════════════════════════════════════════════════════════
# We already have SP→CBS distances. We need APL→CBS distances.
# Run Dijkstra from each APL candidate's nearest graph node.

print("\nBuilding road graph for APL distance computation …")
nodes = pd.read_excel(xl, "Nodes")
edges = pd.read_excel(xl, "Edges")

adj = {int(n): [] for n in nodes["NODE ID"]}
for _, r in edges.iterrows():
    v1, v2, d, ow = int(r["V1"]), int(r["V2"]), float(r["DIST"]), bool(r["ONE_WAY"])
    adj[v1].append((v2, d))
    if not ow:
        adj[v2].append((v1, d))

node_ids    = nodes["NODE ID"].astype(int).tolist()
node_coords = nodes[["X", "Y"]].values.astype(float)
tree        = KDTree(node_coords)

# Coordinate offset: local → RD New
X_OFF, Y_OFF = 128904.25, 270490.225

def snap(coords_2d):
    """Return nearest graph node IDs for array of (x,y) local coords."""
    _, idx = tree.query(np.array(coords_2d, dtype=float))
    return [node_ids[i] for i in (idx if hasattr(idx, "__iter__") else [idx])]

cbs_nodes = snap(cbs[["x", "y"]].values)

def dijkstra(src, adj):
    dist = {src: 0.0}
    pq   = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# APL distance matrix: shape (n_cbs, n_apl)
print(f"  Running Dijkstra from {len(apl_cands)} APL candidates …")
apl_nodes = snap(apl_cands[["x", "y"]].values)
apl_dist_matrix_km = np.full((len(cbs), len(apl_cands)), np.nan)

for k, (apl_id, src_node) in enumerate(zip(apl_cands["square"], apl_nodes)):
    dmap = dijkstra(src_node, adj)
    for i, cn in enumerate(cbs_nodes):
        m = dmap.get(cn, np.nan)
        apl_dist_matrix_km[i, k] = m / 1000.0 if not np.isnan(m) else 999.0
    if (k + 1) % 20 == 0 or k == len(apl_cands) - 1:
        print(f"    {k+1}/{len(apl_cands)} done")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LOAD SP DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading SP distance matrix …")
dist_full      = pd.read_csv(DIST_MATRIX_PATH, index_col=0)
sp_dist_matrix = dist_full.reindex(cbs["square"]).fillna(999).values  # (n_cbs, n_sp)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  COMBINED FACILITY SET
# ═══════════════════════════════════════════════════════════════════════════════
# Facilities: [SP_0, SP_1, …, SP_34,  APL_0, APL_1, …, APL_K]
# Stacking distance matrices horizontally: shape (n_cbs, n_sp + n_apl)

n_sp  = len(sp_df)
n_apl = len(apl_cands)
n_i   = len(cbs)

dist_all = np.hstack([sp_dist_matrix, apl_dist_matrix_km])  # (n_cbs, n_sp+n_apl)

# Facility metadata arrays (length n_sp + n_apl)
fac_type       = ["SP"]  * n_sp  + ["APL"] * n_apl
fac_fixed_cost = [FIXED_COST_SP] * n_sp + [FIXED_COST_APL] * n_apl
fac_capacity   = [SP_STORAGE_CAPACITY] * n_sp + [APL_STORAGE_CAPACITY] * n_apl
fac_ids        = sp_df["sp_id"].astype(str).tolist() + apl_cands["square"].tolist()

n_fac = n_sp + n_apl
print(f"\nFacility pool: {n_sp} SPs + {n_apl} APL candidates = {n_fac} total")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DEMAND & COST MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing cost matrices …")

cbs["pop_frac"]        = cbs["Population"] / cbs["Population"].sum()
cbs["annual_parcels"]  = cbs["pop_frac"] * TOTAL_ANNUAL_PARCELS
cbs["avg_daily_parcels"] = cbs["annual_parcels"] / 365.0

# Pickup probability: APL columns get the boost
p_base = 1.0 / (1.0 + PICKUP_DECAY * dist_all)            # (n_cbs, n_fac)
apl_boost = np.array([1.0] * n_sp + [1.0 + APL_PICKUP_BOOST] * n_apl)
p_pickup = np.minimum(1.0, p_base * apl_boost)             # cap at 1.0

# Annual pickups and deliveries per (zone, facility) pair
ann_parcels_col   = cbs["annual_parcels"].values[:, None]  # (n_cbs, 1)
ann_pickups_all   = ann_parcels_col * p_pickup             # (n_cbs, n_fac)
ann_deliveries_all = ann_parcels_col * (1.0 - p_pickup)   # (n_cbs, n_fac)

# Cost matrix c[i,f] = delivery cost + storage cost - pickup savings
STORAGE_COST_PKG   = STORAGE_COST_PKG_DAY * AVG_STORAGE_DAYS
NET_PICKUP_BENEFIT = DELIVERY_SAVES - STORAGE_COST_PKG

c = (DELIVERY_COST_KM * dist_all * ann_deliveries_all
     - NET_PICKUP_BENEFIT * ann_pickups_all)               # (n_cbs, n_fac)

print(f"  Net pickup benefit    : €{NET_PICKUP_BENEFIT:.2f} / pickup")
print(f"  APL avg p_pickup boost: +{APL_PICKUP_BOOST*100:.0f}% vs SP at same distance")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  ILP FORMULATION
# ═══════════════════════════════════════════════════════════════════════════════
I = list(range(n_i))
F = list(range(n_fac))
SP_F  = list(range(n_sp))           # facility indices that are SPs
APL_F = list(range(n_sp, n_fac))    # facility indices that are APLs

print(f"\nBuilding ILP ({n_i} zones × {n_fac} facilities = {n_i*n_fac:,} assignment vars) …")

model = pulp.LpProblem("Hybrid_SP_APL", pulp.LpMinimize)

# Decision variables
y = pulp.LpVariable.dicts("open",   F,           cat="Binary")
x = pulp.LpVariable.dicts("assign", [(i,f) for i in I for f in F], cat="Binary")

# ── Objective ─────────────────────────────────────────────────────────────────
fixed_part = pulp.lpSum(fac_fixed_cost[f] * y[f] for f in F)
var_part   = pulp.lpSum(c[i, f] * x[(i, f)] for i in I for f in F)
model     += fixed_part + var_part, "Total_Annual_Cost"

# ── C1: each zone assigned to exactly one facility ────────────────────────────
for i in I:
    model += pulp.lpSum(x[(i, f)] for f in F) == 1, f"assign_{i}"

# ── C2: only assign to open facility ──────────────────────────────────────────
for i in I:
    for f in F:
        model += x[(i, f)] <= y[f], f"open_{i}_{f}"

# ── C3: peak storage capacity per facility ────────────────────────────────────
for f in F:
    peak_load = pulp.lpSum(
        (cbs.iloc[i]["avg_daily_parcels"] * p_pickup[i, f]
         * PEAK_RATIO * AVG_STORAGE_DAYS) * x[(i, f)]
        for i in I
    )
    model += peak_load <= fac_capacity[f] * y[f], f"cap_{f}"

# ── C4: optional SP/APL count bounds ──────────────────────────────────────────
if SP_MIN_OPEN is not None:
    model += pulp.lpSum(y[f] for f in SP_F) >= SP_MIN_OPEN, "sp_min"
if SP_MAX_OPEN is not None:
    model += pulp.lpSum(y[f] for f in SP_F) <= SP_MAX_OPEN, "sp_max"
if APL_MAX_OPEN is not None:
    model += pulp.lpSum(y[f] for f in APL_F) <= APL_MAX_OPEN, "apl_max"

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SOLVE
# ═══════════════════════════════════════════════════════════════════════════════
print("Solving with CBC …")
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=SOLVER_TIME_LIMIT)
model.solve(solver)

print(f"\nSolver status : {pulp.LpStatus[model.status]}")
print(f"Objective     : €{pulp.value(model.objective):,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  EXTRACT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
open_f     = [f for f in F if pulp.value(y[f]) > 0.5]
open_sp_f  = [f for f in open_f if fac_type[f] == "SP"]
open_apl_f = [f for f in open_f if fac_type[f] == "APL"]

open_sp_ids  = [fac_ids[f] for f in open_sp_f]
open_apl_ids = [fac_ids[f] for f in open_apl_f]

print(f"\nOpen SPs  ({len(open_sp_f)}/{n_sp})  : {open_sp_ids}")
print(f"Open APLs ({len(open_apl_f)}/{n_apl}) : {open_apl_ids}")

# Assignments
assignments = []
for i in I:
    for f in F:
        if pulp.value(x[(i, f)]) > 0.5:
            assignments.append({
                "cbs_square"        : cbs.iloc[i]["square"],
                "population"        : cbs.iloc[i]["Population"],
                "assigned_facility" : fac_ids[f],
                "facility_type"     : fac_type[f],
                "dist_km"           : dist_all[i, f],
                "pickup_probability": round(p_pickup[i, f], 4),
                "annual_pickups"    : round(ann_pickups_all[i, f]),
                "annual_deliveries" : round(ann_deliveries_all[i, f]),
                "delivery_cost"     : round(DELIVERY_COST_KM * dist_all[i,f] * ann_deliveries_all[i,f], 2),
                "storage_cost"      : round(STORAGE_COST_PKG * ann_pickups_all[i,f], 2),
                "pickup_savings"    : round(DELIVERY_SAVES * ann_pickups_all[i,f], 2),
                "net_variable_cost" : round(c[i, f], 2),
            })
            break

results_df = pd.DataFrame(assignments)

# Facility summary
fac_summary = results_df.groupby(["assigned_facility", "facility_type"]).agg(
    n_zones           = ("cbs_square",        "count"),
    total_population  = ("population",         "sum"),
    avg_dist_km       = ("dist_km",            "mean"),
    avg_pickup_prob   = ("pickup_probability", "mean"),
    annual_pickups    = ("annual_pickups",     "sum"),
    annual_deliveries = ("annual_deliveries",  "sum"),
    delivery_cost     = ("delivery_cost",      "sum"),
    storage_cost      = ("storage_cost",       "sum"),
    pickup_savings    = ("pickup_savings",     "sum"),
    net_var_cost      = ("net_variable_cost",  "sum"),
).reset_index()

fac_summary["fixed_cost"] = fac_summary["facility_type"].map(
    {"SP": FIXED_COST_SP, "APL": FIXED_COST_APL}
)
fac_summary["total_cost"] = fac_summary["net_var_cost"] + fac_summary["fixed_cost"]
fac_summary["avg_daily_pickups"] = fac_summary["annual_pickups"] / 365
fac_summary["peak_storage_load"] = (fac_summary["avg_daily_pickups"]
                                    * PEAK_RATIO * AVG_STORAGE_DAYS)
cap_map = {"SP": SP_STORAGE_CAPACITY, "APL": APL_STORAGE_CAPACITY}
fac_summary["storage_util"] = fac_summary.apply(
    lambda r: r["peak_storage_load"] / cap_map[r["facility_type"]], axis=1
)
fac_summary["bounce_risk"] = fac_summary["storage_util"] > (1 - SP_BOUNCE_THRESHOLD)
fac_summary = fac_summary.sort_values("total_cost", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  REPORT
# ═══════════════════════════════════════════════════════════════════════════════
sp_rows  = results_df[results_df["facility_type"] == "SP"]
apl_rows = results_df[results_df["facility_type"] == "APL"]

total_fixed    = (len(open_sp_f) * FIXED_COST_SP + len(open_apl_f) * FIXED_COST_APL)
total_delivery = results_df["delivery_cost"].sum()
total_storage  = results_df["storage_cost"].sum()
total_savings  = results_df["pickup_savings"].sum()
total_net      = total_fixed + total_delivery + total_storage - total_savings

print("\n" + "="*70)
print("HYBRID SP + APL MODEL — RESULTS SUMMARY")
print("="*70)
print(f"  Open SPs                    : {len(open_sp_f):>3} / {n_sp}")
print(f"  Open APLs                   : {len(open_apl_f):>3} / {n_apl}")
print(f"  Total open facilities       : {len(open_f):>3}")
print()
print(f"  Fixed costs (SPs)           : €{len(open_sp_f)*FIXED_COST_SP:>12,.0f}")
print(f"  Fixed costs (APLs)          : €{len(open_apl_f)*FIXED_COST_APL:>12,.0f}")
print(f"  Delivery costs              : €{total_delivery:>12,.0f}")
print(f"  Storage costs               : €{total_storage:>12,.0f}")
print(f"  Pickup savings              : −€{total_savings:>11,.0f}")
print(f"  {'─'*45}")
print(f"  Total effective cost        : €{total_net:>12,.0f}")
print()
pop_w = (results_df["dist_km"] * results_df["population"]).sum() / results_df["population"].sum()
print(f"  Pop-weighted avg distance   : {pop_w:.3f} km")
print(f"  Avg pickup probability      : {results_df['pickup_probability'].mean():.3f}")
print(f"  Zones served by SP          : {len(sp_rows)}  ({len(sp_rows)/n_i*100:.0f}%)")
print(f"  Zones served by APL         : {len(apl_rows)}  ({len(apl_rows)/n_i*100:.0f}%)")
print(f"  Annual pickups (total)      : {results_df['annual_pickups'].sum():>12,.0f}")
print(f"  Annual deliveries (total)   : {results_df['annual_deliveries'].sum():>12,.0f}")

# Bounce risk
bouncy = fac_summary[fac_summary["bounce_risk"]]
if len(bouncy):
    print(f"\n  ⚠  Facilities at bounce risk : {bouncy['assigned_facility'].tolist()}")
else:
    print(f"\n  ✓  No facilities exceed bounce threshold")

total_peak = fac_summary["peak_storage_load"].sum()
total_cap  = len(open_sp_f)*SP_STORAGE_CAPACITY + len(open_apl_f)*APL_STORAGE_CAPACITY
print(f"  City-wide peak utilisation  : {total_peak/total_cap*100:.1f}%")

closed_sp_ids = [fac_ids[f] for f in SP_F if f not in open_f]
print(f"\n  Closed SPs  ({len(closed_sp_ids)}): {closed_sp_ids}")

print("\nTop 10 facilities by total cost:")
cols = ["assigned_facility","facility_type","n_zones","total_population",
        "avg_dist_km","avg_pickup_prob","total_cost","peak_storage_load"]
print(fac_summary[cols].head(10).to_string(index=False))

# ── APL-specific insight ───────────────────────────────────────────────────────
if len(open_apl_f) > 0:
    apl_fac = fac_summary[fac_summary["facility_type"] == "APL"]
    print(f"\nAPL details ({len(apl_fac)} open):")
    print(apl_fac[["assigned_facility","n_zones","total_population",
                    "avg_dist_km","avg_pickup_prob","total_cost"]].to_string(index=False))

    extra_pickups = apl_rows["annual_pickups"].sum() - (
        apl_rows["annual_parcels"] if "annual_parcels" in apl_rows.columns
        else apl_rows["annual_pickups"].sum() / (1 + APL_PICKUP_BOOST)
    )
    print(f"\n  APL pickup boost generates ~{APL_PICKUP_BOOST*100:.0f}% more pickups vs equivalent SP")
    print(f"  APL fixed cost saving vs SP : €{(FIXED_COST_SP-FIXED_COST_APL)*len(open_apl_f):,.0f}/yr")

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  SENSITIVITY — vary APL fixed cost assumption
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SENSITIVITY — APL fixed cost assumption (how cheap does an APL need to be?)")
print("="*70)

for apl_cost in [5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000]:
    m2     = pulp.LpProblem(f"Hybrid_apl{apl_cost}", pulp.LpMinimize)
    fc2    = [FIXED_COST_SP]*n_sp + [apl_cost]*n_apl
    y2     = pulp.LpVariable.dicts("open",   F, cat="Binary")
    x2     = pulp.LpVariable.dicts("assign", [(i,f) for i in I for f in F], cat="Binary")
    m2    += pulp.lpSum(fc2[f]*y2[f] for f in F) + pulp.lpSum(c[i,f]*x2[(i,f)] for i in I for f in F)
    for i in I:
        m2 += pulp.lpSum(x2[(i,f)] for f in F) == 1
    for i in I:
        for f in F:
            m2 += x2[(i,f)] <= y2[f]
    for f in F:
        m2 += pulp.lpSum(
            (cbs.iloc[i]["avg_daily_parcels"] * p_pickup[i,f]
             * PEAK_RATIO * AVG_STORAGE_DAYS) * x2[(i,f)]
            for i in I
        ) <= fac_capacity[f] * y2[f]
    m2.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if m2.status == 1:
        n_apl_open = sum(1 for f in APL_F if pulp.value(y2[f]) > 0.5)
        n_sp_open  = sum(1 for f in SP_F  if pulp.value(y2[f]) > 0.5)
        print(f"  APL cost €{apl_cost:>6,} → obj €{pulp.value(m2.objective):>10,.0f}"
              f"  |  {n_sp_open} SPs + {n_apl_open} APLs open")

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
results_df.to_csv(OUTPUT_DIR + "hybrid_zone_assignments.csv",  index=False)
fac_summary.to_csv(OUTPUT_DIR + "hybrid_facility_summary.csv", index=False)

# Save APL locations with coordinates for mapping
if len(open_apl_f) > 0:
    open_apl_squares = [fac_ids[f] for f in open_apl_f]
    apl_map = apl_cands[apl_cands["square"].isin(open_apl_squares)][
        ["square", "x", "y", "Population", "apl_affinity"]
    ].copy()
    apl_map.to_csv(OUTPUT_DIR + "open_apl_locations.csv", index=False)
    print(f"\nSaved open APL locations → open_apl_locations.csv")

print("\nOutputs saved:")
print("  hybrid_zone_assignments.csv  — each CBS zone with facility type and full cost breakdown")
print("  hybrid_facility_summary.csv  — per facility: type, cost, demand, bounce flag")
print("  open_apl_locations.csv       — coordinates of chosen APL locations for mapping")
print("\nDone.")
"""
Post&L Maastricht — MCFLP v2
==============================
Key changes from v1
--------------------
CHANGE 1 — Distance-dependent pickup probability (Section 5)
  Instead of a fixed 64.3% pickup share for every zone, each CBS square now
  gets its own pickup probability based on its road-network distance to the
  assigned SP:
      p_pickup(d) = 1 / (1 + PICKUP_DECAY * d)
  Calibrated so that at d=0.87 km (EDA pop-weighted avg) p ≈ 0.643.
  This means the assignment cost c[i,j] now depends on distance in two ways:
    - More distance → more expensive deliveries
    - More distance → lower pickup share → even more deliveries (fewer pickups)

CHANGE 2 — Demand broken into pickups and deliveries per zone (Section 4)
  Instead of only tracking total parcels, each CBS zone now has:
    annual_parcels[i]   = pop_frac[i] × 1,351,852
    p_pickup[i,j]       = f(dist[i,j])          ← per assignment pair
    annual_pickups[i,j] = annual_parcels[i] × p_pickup[i,j]
    annual_deliveries[i,j] = annual_parcels[i] × (1 - p_pickup[i,j])

CHANGE 3 — Full cost breakdown in the objective (Section 5)
  c[i,j] = delivery_cost + storage_cost - pickup_savings
         = DELIVERY_COST_KM × dist[i,j] × annual_deliveries[i,j]
         + STORAGE_COST_PKG × annual_pickups[i,j]
         - DELIVERY_SAVES   × annual_pickups[i,j]
  The last two terms collapse to -(DELIVERY_SAVES - STORAGE_COST_PKG) × pickups.
  Because p_pickup now depends on j, this is NO LONGER a constant offset —
  it properly incentivises the model to open SPs close to dense zones.

CHANGE 4 — Peak-day capacity constraint (Section 6c)
  Capacity is now based on peak storage load, not avg daily pickups.
  From the EDA: avg peak/avg ratio across SPs = 2.34.
  Stored parcels at any moment = avg_daily_pickups × AVG_STORAGE_DAYS (1.7).
  We constrain peak storage (packages physically at the SP at one time):
      peak_storage[j] = peak_daily_pickups × AVG_STORAGE_DAYS
                      ≤ SP_STORAGE_CAPACITY
  The constraint in the ILP:
      sum_i [ parcels[i] × p_pickup[i,j] / 365 × PEAK_RATIO × AVG_STORAGE_DAYS × x[i,j] ]
      ≤ SP_STORAGE_CAPACITY × y[j]

CHANGE 5 — Bounce-rate constraint (Section 6d)  ← NEW
  City-wide bounce rate < 1%, SP-level bounce rate < 2%.
  A bounce occurs when a package cannot be stored (SP full) and must move.
  Proxy: if peak_storage_load / SP_STORAGE_CAPACITY > BOUNCE_THRESHOLD_SP,
  the SP is at bounce risk. We enforce this via the capacity constraint above.
  We also add a soft-check post-solve that flags any SP exceeding 2% utilisation
  headroom.

Cost parameters
---------------
  Fixed cost per SP      : €50,000 / yr
  Delivery cost          : €1.50 / km / parcel
  Storage cost           : €0.10 / pkg / day (avg 1.7 days → €0.17 / pickup)
  Pickup saves delivery  : €1.01 avoided delivery cost per pickup
  Net pickup benefit     : €1.01 − €0.17 = €0.84 / pickup
"""

import warnings
import numpy as np
import pandas as pd
import pulp

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS  — edit freely
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH        = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/data_Maastricht_2025.xlsx"
DIST_MATRIX_PATH = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/outputs-analysis/network_dist_matrix.csv"
OUTPUT_DIR       = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/outputs-analysis/"

# Cost parameters (from case description)
FIXED_COST_PER_SP    = 50_000   # € / year per open SP
DELIVERY_COST_KM     = 1.50     # € / km / delivery parcel
STORAGE_COST_PKG_DAY = 0.10     # € / package / day in storage
AVG_STORAGE_DAYS     = 1.7      # avg days a pickup sits at the SP
DELIVERY_SAVES       = 1.01     # € delivery cost avoided per pickup

# CHANGE 1: pickup probability decay — calibrated to EDA avg distance 0.87 km → p=0.643
# Solve: 0.643 = 1 / (1 + alpha * 0.87)  →  alpha = (1/0.643 - 1) / 0.87 ≈ 0.641
PICKUP_DECAY = 0.641            # alpha in p(d) = 1 / (1 + alpha * d)

# CHANGE 4: peak capacity — EDA shows avg peak/avg ratio = 2.34 across SPs
PEAK_RATIO           = 2.34     # multiply avg daily pickups to get peak day
SP_STORAGE_CAPACITY  = 500      # max packages stored at one SP at peak (assumption)
                                # Increase to relax; set to None to disable

# Bounce rate thresholds (checked post-solve, not hard constraints)
CITY_BOUNCE_THRESHOLD = 0.01    # 1%  city-wide
SP_BOUNCE_THRESHOLD   = 0.02    # 2%  per SP

# Optional SP count bounds (set None to disable)
MIN_OPEN_SPS  = None
MAX_OPEN_SPS  = None

SOLVER_TIME_LIMIT = 120

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
xl  = pd.ExcelFile(DATA_PATH)

sp  = pd.read_excel(xl, "Service Point Locations").rename(
        columns={"Location ID": "sp_id", "X": "x", "Y": "y",
                 "Containing Square": "square"})

cbs = pd.read_excel(xl, "CBS Squares").rename(columns={"Square": "square"})
cbs = cbs.dropna(subset=["Population", "X", "Y"])
cbs = cbs[cbs["Population"] > 0].reset_index(drop=True)
cbs = cbs.rename(columns={"X": "x", "Y": "y"})

act = pd.read_excel(xl, "Daily Activity").rename(
        columns={"Location ID": "sp_id"})

print(f"  Candidate SPs    : {len(sp)}")
print(f"  CBS demand zones : {len(cbs)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DISTANCE MATRIX  (road network, from Dijkstra)
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading network distance matrix …")
dist_full      = pd.read_csv(DIST_MATRIX_PATH, index_col=0)
dist_matrix_km = dist_full.loc[cbs["square"]].values   # shape (n_cbs, n_sp)

print(f"  Distance matrix  : {dist_matrix_km.shape}  (CBS zones × SPs)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DEMAND PARAMETERS  — CHANGE 2: per-zone parcel estimate split into
#     pickups and deliveries as a function of distance
# ═══════════════════════════════════════════════════════════════════════════════
print("Building demand parameters …")

# Scale total network demand to each CBS zone by population share
total_pop    = cbs["Population"].sum()
cbs["pop_frac"] = cbs["Population"] / total_pop

TOTAL_ANNUAL_PARCELS    = act["Deliveries"].sum() + act["Pickups"].sum()   # 1,351,852
cbs["annual_parcels"]   = cbs["pop_frac"] * TOTAL_ANNUAL_PARCELS

# CHANGE 2: pickup probability is a (n_cbs × n_sp) matrix — depends on distance
# p_pickup[i,j] = 1 / (1 + PICKUP_DECAY * dist[i,j])
p_pickup_matrix = 1.0 / (1.0 + PICKUP_DECAY * dist_matrix_km)   # shape (n_cbs, n_sp)

# Annual pickups and deliveries per (zone, SP) pair
# If zone i is assigned to SP j:
annual_parcels_col = cbs["annual_parcels"].values[:, None]         # (n_cbs, 1)
annual_pickups_ij  = annual_parcels_col * p_pickup_matrix          # (n_cbs, n_sp)
annual_deliveries_ij = annual_parcels_col * (1 - p_pickup_matrix)  # (n_cbs, n_sp)

# Avg daily parcels per zone (used in capacity constraint)
# Use avg across all SPs as a zone-level estimate — exact split determined by assignment
cbs["avg_daily_parcels"] = cbs["annual_parcels"] / 365.0

# Sanity check: city-wide averages should match EDA
avg_p = p_pickup_matrix.mean()
print(f"  Avg pickup probability across all (zone, SP) pairs : {avg_p:.3f}  (EDA: 0.643)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ASSIGNMENT COST MATRIX  c[i,j]  — CHANGE 3: full cost breakdown
# ═══════════════════════════════════════════════════════════════════════════════
# Cost when zone i is assigned to SP j:
#
#   delivery_cost[i,j]   = DELIVERY_COST_KM × dist[i,j] × annual_deliveries[i,j]
#   storage_cost[i,j]    = STORAGE_COST_PKG_DAY × AVG_STORAGE_DAYS × annual_pickups[i,j]
#   pickup_saving[i,j]   = DELIVERY_SAVES × annual_pickups[i,j]
#
#   c[i,j] = delivery_cost + storage_cost - pickup_saving
#           = DELIVERY_COST_KM × dist × (1 - p_pickup) × parcels
#             + (STORAGE_COST_PKG_DAY × AVG_STORAGE_DAYS - DELIVERY_SAVES) × p_pickup × parcels

STORAGE_COST_PKG = STORAGE_COST_PKG_DAY * AVG_STORAGE_DAYS   # = 0.17 €/pickup
NET_PICKUP_BENEFIT = DELIVERY_SAVES - STORAGE_COST_PKG        # = 0.84 €/pickup

delivery_cost_matrix = DELIVERY_COST_KM * dist_matrix_km * annual_deliveries_ij
pickup_net_matrix    = -NET_PICKUP_BENEFIT * annual_pickups_ij   # negative = savings

c = delivery_cost_matrix + pickup_net_matrix   # shape (n_cbs, n_sp)

print(f"  Total annual cost estimate (full assignment): €{c.sum():>14,.0f}")
print(f"  Net pickup benefit used    : €{NET_PICKUP_BENEFIT:.2f} per pickup")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ILP FORMULATION
# ═══════════════════════════════════════════════════════════════════════════════
n_i    = len(cbs)
n_j    = len(sp)
sp_ids = sp["sp_id"].tolist()
I      = list(range(n_i))
J      = list(range(n_j))

print(f"\nBuilding ILP  ({n_i} zones × {n_j} SPs = {n_i*n_j:,} assignment vars) …")

model = pulp.LpProblem("MCFLP_v2_PostL", pulp.LpMinimize)

y = pulp.LpVariable.dicts("open",   J,           cat="Binary")
x = pulp.LpVariable.dicts("assign", [(i,j) for i in I for j in J], cat="Binary")

# ── 6a. Objective ─────────────────────────────────────────────────────────────
fixed_costs      = FIXED_COST_PER_SP * pulp.lpSum(y[j] for j in J)
assignment_costs = pulp.lpSum(c[i,j] * x[(i,j)] for i in I for j in J)
model           += fixed_costs + assignment_costs, "Total_Annual_Cost"

# ── 6b. C1: every zone assigned to exactly one open SP ────────────────────────
for i in I:
    model += pulp.lpSum(x[(i,j)] for j in J) == 1, f"assign_{i}"

# ── 6c. C2: only assign to an open SP ─────────────────────────────────────────
for i in I:
    for j in J:
        model += x[(i,j)] <= y[j], f"open_{i}_{j}"

# ── 6d. C3: peak storage capacity — CHANGE 4 ──────────────────────────────────
# Peak packages stored at SP j on the busiest day:
#   sum_i [ (annual_parcels[i] / 365) × p_pickup[i,j] × PEAK_RATIO × AVG_STORAGE_DAYS × x[i,j] ]
#   ≤ SP_STORAGE_CAPACITY × y[j]
#
# This replaces the old avg-daily-pickups constraint and ties to the bounce risk.
if SP_STORAGE_CAPACITY is not None:
    for j in J:
        peak_load_j = pulp.lpSum(
            (cbs.iloc[i]["avg_daily_parcels"] * p_pickup_matrix[i, j]
             * PEAK_RATIO * AVG_STORAGE_DAYS) * x[(i,j)]
            for i in I
        )
        model += peak_load_j <= SP_STORAGE_CAPACITY * y[j], f"peak_cap_{j}"

# ── 6e. C4: optional SP count bounds ──────────────────────────────────────────
if MIN_OPEN_SPS is not None:
    model += pulp.lpSum(y[j] for j in J) >= MIN_OPEN_SPS, "min_sps"
if MAX_OPEN_SPS is not None:
    model += pulp.lpSum(y[j] for j in J) <= MAX_OPEN_SPS, "max_sps"

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SOLVE
# ═══════════════════════════════════════════════════════════════════════════════
print("Solving with CBC …")
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=SOLVER_TIME_LIMIT)
model.solve(solver)

status = pulp.LpStatus[model.status]
print(f"\nSolver status : {status}")
print(f"Objective     : €{pulp.value(model.objective):,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  EXTRACT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
open_j      = [j for j in J if pulp.value(y[j]) > 0.5]
open_sp_ids = [sp_ids[j] for j in open_j]
print(f"\nOpen SPs ({len(open_j)}) : {open_sp_ids}")

assignments = []
for i in I:
    for j in J:
        if pulp.value(x[(i,j)]) > 0.5:
            p_pick   = p_pickup_matrix[i, j]
            ann_pkup = annual_pickups_ij[i, j]
            ann_del  = annual_deliveries_ij[i, j]
            assignments.append({
                "cbs_square"           : cbs.iloc[i]["square"],
                "population"           : cbs.iloc[i]["Population"],
                "assigned_sp"          : sp_ids[j],
                "dist_km"              : dist_matrix_km[i, j],
                "pickup_probability"   : round(p_pick, 4),      # CHANGE 1
                "annual_pickups"       : round(ann_pkup),        # CHANGE 2
                "annual_deliveries"    : round(ann_del),         # CHANGE 2
                "delivery_cost"        : round(DELIVERY_COST_KM * dist_matrix_km[i,j] * ann_del, 2),
                "storage_cost"         : round(STORAGE_COST_PKG * ann_pkup, 2),
                "pickup_savings"       : round(DELIVERY_SAVES * ann_pkup, 2),
                "net_assignment_cost"  : round(c[i,j], 2),      # CHANGE 3
            })
            break

results_df = pd.DataFrame(assignments)

# SP-level summary
sp_summary = results_df.groupby("assigned_sp").agg(
    n_zones           = ("cbs_square",        "count"),
    total_population  = ("population",         "sum"),
    avg_dist_km       = ("dist_km",            "mean"),
    avg_pickup_prob   = ("pickup_probability", "mean"),   # CHANGE 1
    annual_pickups    = ("annual_pickups",     "sum"),    # CHANGE 2
    annual_deliveries = ("annual_deliveries",  "sum"),    # CHANGE 2
    delivery_cost     = ("delivery_cost",      "sum"),
    storage_cost      = ("storage_cost",       "sum"),
    pickup_savings    = ("pickup_savings",     "sum"),
    net_variable_cost = ("net_assignment_cost","sum"),
).reset_index()
sp_summary["fixed_cost"]  = FIXED_COST_PER_SP
sp_summary["total_cost"]  = sp_summary["net_variable_cost"] + FIXED_COST_PER_SP

# CHANGE 4: peak storage load per SP
sp_summary["avg_daily_pickups"] = sp_summary["annual_pickups"] / 365
sp_summary["peak_storage_load"] = sp_summary["avg_daily_pickups"] * PEAK_RATIO * AVG_STORAGE_DAYS
sp_summary["storage_utilisation"] = sp_summary["peak_storage_load"] / SP_STORAGE_CAPACITY if SP_STORAGE_CAPACITY else np.nan

# CHANGE 5: bounce risk flag
sp_summary["bounce_risk"] = sp_summary["storage_utilisation"] > (1 - SP_BOUNCE_THRESHOLD)

sp_summary = sp_summary.sort_values("total_cost", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  REPORT
# ═══════════════════════════════════════════════════════════════════════════════
total_fixed    = len(open_j) * FIXED_COST_PER_SP
total_delivery = results_df["delivery_cost"].sum()
total_storage  = results_df["storage_cost"].sum()
total_savings  = results_df["pickup_savings"].sum()
total_net      = total_fixed + total_delivery + total_storage - total_savings

print("\n" + "="*65)
print("RESULTS SUMMARY  (v2 — distance-dependent pickup probability)")
print("="*65)
print(f"  Open SPs                  : {len(open_j)} / {n_j}")
print(f"  Fixed costs               : €{total_fixed:>12,.0f}")
print(f"  Delivery costs            : €{total_delivery:>12,.0f}")
print(f"  Storage costs             : €{total_storage:>12,.0f}")
print(f"  Pickup savings            : −€{total_savings:>11,.0f}")
print(f"  ──────────────────────────────────────────")
print(f"  Total effective cost      : €{total_net:>12,.0f}")

pop_w = (results_df["dist_km"]*results_df["population"]).sum() / results_df["population"].sum()
avg_p_actual = results_df["pickup_probability"].mean()
print(f"\n  Pop-weighted avg distance : {pop_w:.3f} km")
print(f"  Avg pickup probability    : {avg_p_actual:.3f}  (city avg @ this solution)")
print(f"  Annual pickups            : {results_df['annual_pickups'].sum():,.0f}")
print(f"  Annual deliveries         : {results_df['annual_deliveries'].sum():,.0f}")

closed_ids = [sid for sid in sp_ids if sid not in open_sp_ids]
print(f"\n  Closed SPs ({len(closed_ids)})  : {closed_ids}")

# CHANGE 5: Bounce-risk check
bouncy = sp_summary[sp_summary["bounce_risk"] == True]
if len(bouncy) > 0:
    print(f"\n  ⚠  SPs at bounce risk (utilisation > {(1-SP_BOUNCE_THRESHOLD)*100:.0f}%): {bouncy['assigned_sp'].tolist()}")
else:
    print(f"\n  ✓  No SPs exceed SP bounce threshold ({SP_BOUNCE_THRESHOLD*100:.0f}%)")

# City-level bounce proxy: total demand vs total capacity
total_peak_storage = sp_summary["peak_storage_load"].sum()
total_capacity     = len(open_j) * SP_STORAGE_CAPACITY if SP_STORAGE_CAPACITY else np.nan
city_util          = total_peak_storage / total_capacity if SP_STORAGE_CAPACITY else np.nan
print(f"  City-wide peak utilisation: {city_util*100:.1f}%  (threshold: {CITY_BOUNCE_THRESHOLD*100:.0f}%)")

print("\nTop 10 SPs by total cost:")
cols = ["assigned_sp","n_zones","total_population","avg_dist_km",
        "avg_pickup_prob","annual_pickups","annual_deliveries","total_cost","peak_storage_load"]
print(sp_summary[cols].head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  SENSITIVITY — vary number of open SPs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("SENSITIVITY — number of open SPs")
print("="*65)

sens_results = []
n_opt = len(open_j)
for n_open in range(max(5, n_opt-4), min(n_j+1, n_opt+6)):
    m2 = pulp.LpProblem(f"MCFLP_v2_n{n_open}", pulp.LpMinimize)
    y2 = pulp.LpVariable.dicts("open",   J, cat="Binary")
    x2 = pulp.LpVariable.dicts("assign", [(i,j) for i in I for j in J], cat="Binary")
    m2 += FIXED_COST_PER_SP * pulp.lpSum(y2[j] for j in J) + \
          pulp.lpSum(c[i,j]*x2[(i,j)] for i in I for j in J)
    for i in I:
        m2 += pulp.lpSum(x2[(i,j)] for j in J) == 1
    for i in I:
        for j in J:
            m2 += x2[(i,j)] <= y2[j]
    if SP_STORAGE_CAPACITY:
        for j in J:
            m2 += pulp.lpSum(
                (cbs.iloc[i]["avg_daily_parcels"] * p_pickup_matrix[i,j]
                 * PEAK_RATIO * AVG_STORAGE_DAYS) * x2[(i,j)]
                for i in I
            ) <= SP_STORAGE_CAPACITY * y2[j]
    m2 += pulp.lpSum(y2[j] for j in J) == n_open
    m2.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if m2.status == 1:
        sens_results.append({"n_open": n_open, "objective": pulp.value(m2.objective)})
        print(f"  n={n_open:2d} → €{pulp.value(m2.objective):>12,.0f}")

sens_df = pd.DataFrame(sens_results)

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
results_df.to_csv(OUTPUT_DIR + "zone_assignments_v2.csv",  index=False)
sp_summary.to_csv(OUTPUT_DIR + "sp_summary_v2.csv",        index=False)
sens_df.to_csv(   OUTPUT_DIR + "sensitivity_v2.csv",       index=False)

print("\nOutputs saved:")
print("  zone_assignments_v2.csv  — per CBS zone: pickup prob, split, costs")
print("  sp_summary_v2.csv        — per SP: demand split, cost breakdown, bounce flag")
print("  sensitivity_v2.csv       — objective vs number of open SPs")
print("\nDone.")
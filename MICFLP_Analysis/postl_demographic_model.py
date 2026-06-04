"""
postl_demographic_model.py
===========================
Extends the Hybrid SP + APL model (v2) with two demographic improvements:

IMPROVEMENT 1 — Demographic demand weighting
  Instead of scaling demand purely by population, each CBS square gets a
  demand index that reflects how much online ordering that population does:

    demand_index[i] =
        1.20 × Age25-44 share  (peak online shoppers)
      + 1.10 × Age15-24 share  (high online, mostly pickup)
      + 1.00 × Age45-64 share  (average)
      + 0.75 × Age65+   share  (below average online ordering)
      + 0.40 × Age0-14  share  (not ordering themselves)
      — all multiplied by population, then scaled by income

  Weights are assumptions — defensible based on e-commerce research.
  Validated below by checking correlation with actual SP volumes.

IMPROVEMENT 2 — Age-adjusted pickup probability decay
  The base pickup probability p(d) = 1/(1+0.639*d) now gets multiplied
  by an age factor per CBS square:

    pickup_age_factor[i] =
        young_share[i] × YOUNG_PICKUP_FACTOR   (more likely to pickup: 24/7 APL suits them)
      + middle_share[i] × 1.0                  (baseline behaviour)
      + elderly_share[i] × ELDERLY_PICKUP_FACTOR (less likely to pickup: prefer home delivery)

  Key assumptions:
    YOUNG_PICKUP_FACTOR  = 1.25  (25% more likely to pickup than average)
    ELDERLY_PICKUP_FACTOR = 0.65 (35% less likely to pickup — elderly prefer delivery)

  This means:
    - Young-dominated squares assigned to APLs benefit more (APL 24/7 + young boost)
    - Elderly-dominated squares generate more deliveries regardless of facility distance
    - The model naturally places APLs near young populations (higher pickup savings)

  Results in three models compared side-by-side:
    Model A — Baseline (all 35 SPs, nearest-SP assignment, no demographics)
    Model B — Hybrid SP+APL (v2, uniform demand and pickup probability)
    Model C — Demographic Hybrid (this model, age-weighted demand + age pickup factor)

Cost parameters (unchanged from v2)
-------------------------------------
  Fixed cost SP : €50,000 / yr
  Fixed cost APL: €15,000 / yr
  Delivery      : €1.50 / km / delivery parcel
  Storage       : €0.17 / pickup (€0.10/day × 1.7 days)
  Pickup saves  : €1.01 / pickup → net benefit €0.84/pickup
  Peak ratio    : 2.34 × avg (from EDA)
  SP capacity   : 500 packages peak
  APL capacity  : 120 packages peak
"""

import warnings
import heapq
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import pulp

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/data_Maastricht_2025.xlsx"
DIST_PATH    = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/outputs-analysis/network_dist_matrix.csv"
OUTPUT_DIR   = "/Users/beliz/Desktop/business_analytics/year2/E-Lab 2 Cases/Post&L-eLab2/outputs-analysis/"

# Cost parameters
FIXED_SP         = 50_000
FIXED_APL        = 15_000
DELIVERY_KM      = 1.50
STORAGE_PKG      = 0.10 * 1.7   # €0.17 per pickup
DELIVERY_SAVES   = 1.01
NET_PU_BENEFIT   = DELIVERY_SAVES - STORAGE_PKG   # €0.84
PICKUP_DECAY     = 0.639         # calibrated to EDA avg 0.87km → 64.3%
APL_BOOST        = 0.15          # APL 24/7 pickup probability boost
PEAK_RATIO       = 2.34
SP_CAP           = 500
APL_CAP          = 120
SOLVER_LIMIT     = 180

# ── DEMOGRAPHIC ASSUMPTIONS  ← key assumptions to document ──────────────────

# Demand weights by age group (relative to average adult)
# Young adults (15-44) are the heaviest online shoppers
# Children (0-14) do not order themselves
# Elderly (65+) order less frequently online
DEMAND_WEIGHT_AGE0_14  = 0.40   # children — parcel demand via parents, low
DEMAND_WEIGHT_AGE15_24 = 1.10   # students/young adults — high online ordering
DEMAND_WEIGHT_AGE25_44 = 1.20   # peak online shoppers — highest weight
DEMAND_WEIGHT_AGE45_64 = 1.00   # baseline
DEMAND_WEIGHT_AGE65    = 0.75   # below average — less online, prefer in-store

# Income multiplier — higher income → more online spending
INCOME_MULTIPLIER = {
    "00-20 low":                    0.75,
    "00-40 laag tot onder midden":  0.80,
    "20-40 below middle":           0.85,
    "20-60 below middle to middle": 0.90,
    "40-60 midden":                 1.00,
    "40-80 middle to above middle": 1.10,
    "60-80 above middle":           1.15,
    "60-100 above middle to high":  1.20,
    "80-100 high":                  1.30,
}

# Pickup probability age adjustment
# Young adults are more willing to go collect (especially from 24/7 APLs)
# Elderly prefer home delivery — lower pickup probability regardless of distance
YOUNG_PICKUP_FACTOR   = 1.25   # Age15-44: 25% more likely to pickup
MIDDLE_PICKUP_FACTOR  = 1.00   # Age45-64: baseline
ELDERLY_PICKUP_FACTOR = 0.65   # Age65+:   35% less likely to pickup

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
xl    = pd.ExcelFile(DATA_PATH)
sp_df = pd.read_excel(xl, "Service Point Locations").rename(
            columns={"Location ID":"sp_id","X":"x","Y":"y","Containing Square":"square"})
cbs   = pd.read_excel(xl, "CBS Squares").rename(columns={"Square":"square","X":"x","Y":"y"})
cbs   = cbs.dropna(subset=["Population","x","y"])
cbs   = cbs[cbs["Population"] > 0].reset_index(drop=True)
act   = pd.read_excel(xl, "Daily Activity").rename(columns={"Location ID":"sp_id"})
nodes = pd.read_excel(xl, "Nodes")
edges = pd.read_excel(xl, "Edges")

TOTAL_PARCELS = act["Deliveries"].sum() + act["Pickups"].sum()
print(f"  CBS zones: {len(cbs)}  |  SPs: {len(sp_df)}  |  Total parcels: {TOTAL_PARCELS:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DEMOGRAPHIC DEMAND INDEX  (Improvement 1)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding demographic demand index …")

def safe_share(col, pop):
    return col.fillna(0) / pop.replace(0, np.nan).fillna(1)

cbs["share_0_14"]  = safe_share(cbs["Age0-14"],  cbs["Population"])
cbs["share_15_24"] = safe_share(cbs["Age15-24"], cbs["Population"])
cbs["share_25_44"] = safe_share(cbs["Age25-44"], cbs["Population"])
cbs["share_45_64"] = safe_share(cbs["Age45-64"], cbs["Population"])
cbs["share_65"]    = safe_share(cbs["Age65+"],   cbs["Population"])

# Weighted age demand score per person in each zone
cbs["age_demand_score"] = (
    DEMAND_WEIGHT_AGE0_14  * cbs["share_0_14"]  +
    DEMAND_WEIGHT_AGE15_24 * cbs["share_15_24"] +
    DEMAND_WEIGHT_AGE25_44 * cbs["share_25_44"] +
    DEMAND_WEIGHT_AGE45_64 * cbs["share_45_64"] +
    DEMAND_WEIGHT_AGE65    * cbs["share_65"]
)

# Income multiplier
cbs["income_mult"] = cbs["Median household income"].map(INCOME_MULTIPLIER).fillna(1.0)

# Final demand index: age-weighted population × income multiplier
cbs["demand_index"] = cbs["Population"] * cbs["age_demand_score"] * cbs["income_mult"]

# Normalise to population fractions (demand_frac sums to 1)
cbs["uniform_frac"]     = cbs["Population"]     / cbs["Population"].sum()
cbs["demographic_frac"] = cbs["demand_index"]   / cbs["demand_index"].sum()

# Annual parcel demand per zone under each weighting
cbs["annual_parcels_uniform"] = cbs["uniform_frac"]     * TOTAL_PARCELS
cbs["annual_parcels_demog"]   = cbs["demographic_frac"] * TOTAL_PARCELS
cbs["avg_daily_demog"]        = cbs["annual_parcels_demog"] / 365.0

# Show which zones changed most
cbs["demand_shift"] = (cbs["demographic_frac"] / cbs["uniform_frac"]).fillna(1.0)
high_demand = cbs.nlargest(5, "demand_shift")[["square","Population","share_25_44","income_mult","demand_shift"]]
low_demand  = cbs.nsmallest(5, "demand_shift")[["square","Population","share_65","income_mult","demand_shift"]]
print(f"\n  Top 5 zones getting HIGHER demand weight (young/high-income):")
print(high_demand.to_string(index=False))
print(f"\n  Top 5 zones getting LOWER demand weight (elderly/low-income):")
print(low_demand.to_string(index=False))

# ── Validation: does demographic index predict actual SP volumes better? ──────
print("\nValidating demand index against actual SP volumes …")
sp_vol = act.groupby("sp_id").agg(
    actual_pickups=("Pickups", "sum"),
    actual_total  =("Pickups","sum")
).reset_index()
sp_vol["actual_total"] = act.groupby("sp_id")[["Pickups","Deliveries"]].sum().sum(axis=1).values

dist_full  = pd.read_csv(DIST_PATH, index_col=0)
sp_ids_str = sp_df["sp_id"].astype(str).tolist()

# For each SP: sum up demand from its nearest CBS zones (proxy for catchment)
cbs_nearest_sp = dist_full.reindex(cbs["square"]).fillna(999).idxmin(axis=1)
cbs["nearest_sp"] = cbs_nearest_sp.astype(int).values

for label, frac_col in [("Uniform pop", "uniform_frac"), ("Demographic", "demographic_frac")]:
    sp_pred = (cbs.groupby("nearest_sp")[frac_col].sum() * TOTAL_PARCELS).reset_index()
    sp_pred.columns = ["sp_id", "predicted"]
    merged = sp_pred.merge(sp_vol[["sp_id","actual_total"]], on="sp_id", how="inner")
    corr = merged["predicted"].corr(merged["actual_total"])
    print(f"  {label:20s} → correlation with actual SP volume: r = {corr:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  AGE-ADJUSTED PICKUP PROBABILITY  (Improvement 2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding age-adjusted pickup factor per CBS zone …")

# Combined young share (15-44) and elderly share (65+)
cbs["young_share"]   = cbs["share_15_24"] + cbs["share_25_44"]
cbs["middle_share"]  = cbs["share_45_64"]
cbs["elderly_share"] = cbs["share_65"]
# Note: children (0-14) use middle factor — their packages are for parents

# Per-zone pickup age factor: weighted average of age group factors
cbs["age_pickup_factor"] = (
    cbs["young_share"]   * YOUNG_PICKUP_FACTOR   +
    cbs["middle_share"]  * MIDDLE_PICKUP_FACTOR  +
    cbs["elderly_share"] * ELDERLY_PICKUP_FACTOR +
    cbs["share_0_14"]    * MIDDLE_PICKUP_FACTOR  # children's parcels go to parents
)

# Clip to reasonable range [0.5, 1.4]
cbs["age_pickup_factor"] = cbs["age_pickup_factor"].clip(0.5, 1.4).fillna(1.0)

print(f"  Age pickup factor range: {cbs['age_pickup_factor'].min():.3f} — {cbs['age_pickup_factor'].max():.3f}")
print(f"  Mean factor: {cbs['age_pickup_factor'].mean():.3f}")

elderly_zones = cbs.nlargest(5,"elderly_share")[["square","Population","elderly_share","age_pickup_factor"]]
young_zones   = cbs.nlargest(5,"young_share")[["square","Population","young_share","age_pickup_factor"]]
print(f"\n  Most elderly zones (lower pickup probability):")
print(elderly_zones.to_string(index=False))
print(f"\n  Youngest zones (higher pickup probability):")
print(young_zones.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ROAD GRAPH + APL CANDIDATES + DISTANCE MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding road graph …")
adj = {int(n): [] for n in nodes["NODE ID"]}
for _, r in edges.iterrows():
    v1,v2,d,ow = int(r["V1"]),int(r["V2"]),float(r["DIST"]),bool(r["ONE_WAY"])
    adj[v1].append((v2,d))
    if not ow: adj[v2].append((v1,d))

node_ids    = nodes["NODE ID"].astype(int).tolist()
node_coords = nodes[["X","Y"]].values.astype(float)
ktree       = KDTree(node_coords)

def snap(coords):
    _, idx = ktree.query(np.array(coords, dtype=float))
    return [node_ids[i] for i in (idx if hasattr(idx,"__iter__") else [idx])]

def dijkstra(src, adj):
    dist = {src: 0.0}; pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")): continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist

cbs_nodes = snap(cbs[["x","y"]].values)

# APL candidates (same affinity filter as v2)
def income_score(val):
    m = {"00-20 low":0.1,"00-40 laag tot onder midden":0.2,"20-40 below middle":0.3,
         "20-60 below middle to middle":0.4,"40-60 midden":0.5,
         "40-80 middle to above middle":0.65,"60-80 above middle":0.75,
         "60-100 above middle to high":0.85,"80-100 high":1.0}
    return m.get(str(val).strip(), 0.5)

sp_squares = set(sp_df["square"].tolist())
apl_raw    = cbs[~cbs["square"].isin(sp_squares)].copy().reset_index(drop=True)
apl_raw["inc_sc"]     = apl_raw["Median household income"].apply(income_score)
apl_raw["ya_share"]   = (apl_raw[["Age15-24","Age25-44"]].sum(axis=1).fillna(0) /
                          apl_raw["Population"].replace(0,np.nan)).fillna(0)
apl_raw["urban_norm"] = (apl_raw["Urbanization index"].fillna(3)-1)/4.0
apl_raw["affinity"]   = 0.4*apl_raw["ya_share"] + 0.3*apl_raw["urban_norm"] + 0.3*apl_raw["inc_sc"]
apl_cands = apl_raw[apl_raw["affinity"] >= apl_raw["affinity"].median()].copy().reset_index(drop=True)

print(f"  APL candidates: {len(apl_cands)}")

# APL distances via Dijkstra
print(f"  Running Dijkstra from {len(apl_cands)} APL candidates …")
apl_nodes = snap(apl_cands[["x","y"]].values)
apl_dist  = np.full((len(cbs), len(apl_cands)), 999.0)
for k, src in enumerate(apl_nodes):
    dmap = dijkstra(src, adj)
    for i, cn in enumerate(cbs_nodes):
        v = dmap.get(cn, np.nan)
        apl_dist[i,k] = v/1000.0 if not np.isnan(v) else 999.0
    if (k+1) % 30 == 0:
        print(f"    {k+1}/{len(apl_cands)} done")

# SP distances
sp_dist = pd.read_csv(DIST_PATH, index_col=0).reindex(cbs["square"]).fillna(999).values

# Combined facility arrays
n_sp  = len(sp_df); n_apl = len(apl_cands)
n_fac = n_sp + n_apl; n_i = len(cbs)
F = list(range(n_fac)); I = list(range(n_i))
SP_F  = list(range(n_sp)); APL_F = list(range(n_sp, n_fac))

dist_all  = np.hstack([sp_dist, apl_dist])
fac_type  = ["SP"]*n_sp  + ["APL"]*n_apl
fac_fixed = [FIXED_SP]*n_sp  + [FIXED_APL]*n_apl
fac_cap   = [SP_CAP]*n_sp    + [APL_CAP]*n_apl
fac_ids   = sp_df["sp_id"].astype(str).tolist() + apl_cands["square"].tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  BUILD COST MATRICES FOR BOTH MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding cost matrices …")

# APL boost array
boost_arr = np.array([1.0]*n_sp + [1.0 + APL_BOOST]*n_apl)

# ── Model B (uniform, no demographics) ───────────────────────────────────────
ann_parcels_B = cbs["annual_parcels_uniform"].values[:,None]
p_B           = np.minimum(1.0, 1.0/(1.0 + PICKUP_DECAY*dist_all) * boost_arr)
ann_pu_B      = ann_parcels_B * p_B
ann_del_B     = ann_parcels_B * (1 - p_B)
c_B           = DELIVERY_KM * dist_all * ann_del_B - NET_PU_BENEFIT * ann_pu_B

# ── Model C (demographic demand + age pickup factor) ─────────────────────────
ann_parcels_C     = cbs["annual_parcels_demog"].values[:,None]
age_factor_col    = cbs["age_pickup_factor"].values[:,None]    # (n_i, 1)

# Base pickup probability × age factor × APL boost
p_C = np.minimum(1.0,
      1.0/(1.0 + PICKUP_DECAY*dist_all) * age_factor_col * boost_arr)

ann_pu_C  = ann_parcels_C * p_C
ann_del_C = ann_parcels_C * (1 - p_C)
c_C       = DELIVERY_KM * dist_all * ann_del_C - NET_PU_BENEFIT * ann_pu_C

# Avg daily parcels (for capacity constraint — use demographic version)
avg_daily_C = cbs["avg_daily_demog"].values

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SOLVE MODEL B  (Hybrid uniform — for comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def solve_model(c_mat, avg_daily, p_mat, label, time_limit=SOLVER_LIMIT):
    print(f"\nSolving {label} …")
    mod = pulp.LpProblem(label.replace(" ","_"), pulp.LpMinimize)
    y   = pulp.LpVariable.dicts("open",   F, cat="Binary")
    x   = pulp.LpVariable.dicts("assign", [(i,f) for i in I for f in F], cat="Binary")

    mod += (pulp.lpSum(fac_fixed[f]*y[f] for f in F) +
            pulp.lpSum(c_mat[i,f]*x[(i,f)] for i in I for f in F))
    for i in I:
        mod += pulp.lpSum(x[(i,f)] for f in F) == 1
    for i in I:
        for f in F:
            mod += x[(i,f)] <= y[f]
    for f in F:
        mod += pulp.lpSum(
            (avg_daily[i] * p_mat[i,f] * PEAK_RATIO * 1.7) * x[(i,f)]
            for i in I
        ) <= fac_cap[f] * y[f]

    mod.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    open_f    = [f for f in F if pulp.value(y[f]) > 0.5]
    open_sp   = [f for f in open_f if fac_type[f]=="SP"]
    open_apl  = [f for f in open_f if fac_type[f]=="APL"]

    assign_dist = np.zeros(n_i)
    assign_type = [""] * n_i
    for i in I:
        for f in F:
            if pulp.value(x[(i,f)]) > 0.5:
                assign_dist[i] = dist_all[i,f]
                assign_type[i] = fac_type[f]
                break

    return {
        "label"       : label,
        "status"      : pulp.LpStatus[mod.status],
        "objective"   : pulp.value(mod.objective),
        "n_sp_open"   : len(open_sp),
        "n_apl_open"  : len(open_apl),
        "open_sp_ids" : [fac_ids[f] for f in open_sp],
        "open_apl_ids": [fac_ids[f] for f in open_apl],
        "assign_dist" : assign_dist,
        "assign_type" : assign_type,
        "model"       : mod,
        "y"           : y,
        "x"           : x,
        "c_mat"       : c_mat,
        "p_mat"       : p_mat,
        "ann_pu"      : ann_pu_C if "Demographic" in label else ann_pu_B,
        "ann_del"     : ann_del_C if "Demographic" in label else ann_del_B,
    }

result_B = solve_model(c_B, cbs["avg_daily"].values if "avg_daily" in cbs.columns
                       else cbs["annual_parcels_uniform"].values/365,
                       p_B, "Model B — Hybrid uniform")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  SOLVE MODEL C  (Demographic hybrid)
# ═══════════════════════════════════════════════════════════════════════════════
cbs["avg_daily"] = cbs["avg_daily_demog"]
result_C = solve_model(c_C, avg_daily_C, p_C, "Model C — Demographic hybrid")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  COMPARE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPARISON: Uniform Hybrid (B) vs Demographic Hybrid (C)")
print("="*70)

for res in [result_B, result_C]:
    ad = res["assign_dist"]
    pop = cbs["Population"].values
    pop_w = (ad * pop).sum() / pop.sum()
    n_apl_zones = sum(1 for t in res["assign_type"] if t=="APL")

    # Cost breakdown
    open_f  = [f for f in F if pulp.value(res["y"][f]) > 0.5]
    t_fixed = sum(fac_fixed[f] for f in open_f)
    t_del   = sum(res["c_mat"][i,f] + NET_PU_BENEFIT * res["p_mat"][i,f] *
                  (res["ann_del"][i,f] + res["ann_pu"][i,f]) / (1 + NET_PU_BENEFIT)
                  for i in I for f in F if pulp.value(res["x"][(i,f)]) > 0.5) - t_fixed

    # Simpler cost breakdown directly
    rows = []
    for i in I:
        for f in F:
            if pulp.value(res["x"][(i,f)]) > 0.5:
                rows.append({
                    "del_cost": DELIVERY_KM * dist_all[i,f] * res["ann_del"][i,f],
                    "sto_cost": 0.10*1.7 * res["ann_pu"][i,f],
                    "pu_save" : DELIVERY_SAVES * res["ann_pu"][i,f],
                    "pickups" : res["ann_pu"][i,f],
                    "deliveries": res["ann_del"][i,f],
                })
                break
    df_r = pd.DataFrame(rows)
    total_fixed    = t_fixed
    total_delivery = df_r["del_cost"].sum()
    total_storage  = df_r["sto_cost"].sum()
    total_savings  = df_r["pu_save"].sum()
    effective      = total_fixed + total_delivery + total_storage - total_savings

    print(f"\n{'─'*60}")
    print(f"  {res['label']}")
    print(f"{'─'*60}")
    print(f"  Status           : {res['status']}")
    print(f"  Open SPs         : {res['n_sp_open']} / {n_sp}")
    print(f"  Open APLs        : {res['n_apl_open']}")
    print(f"  Zones via APL    : {n_apl_zones} / {n_i}")
    print(f"  Model objective  : €{res['objective']:>12,.0f}")
    print(f"  Fixed costs      : €{total_fixed:>12,.0f}")
    print(f"  Delivery costs   : €{total_delivery:>12,.0f}")
    print(f"  Storage costs    : €{total_storage:>12,.0f}")
    print(f"  Pickup savings   : −€{total_savings:>11,.0f}")
    print(f"  Effective total  : €{effective:>12,.0f}")
    print(f"  Pop-wtd avg dist : {pop_w:.3f} km")
    print(f"  Annual pickups   : {df_r['pickups'].sum():>12,.0f}")
    print(f"  Annual deliveries: {df_r['deliveries'].sum():>12,.0f}")
    print(f"  Pickup share     : {df_r['pickups'].sum()/(df_r['pickups'].sum()+df_r['deliveries'].sum())*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  WHERE DID APLs GO? — compare placement between B and C
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("APL PLACEMENT COMPARISON — do demographics shift where APLs open?")
print("="*70)

apl_B = set(result_B["open_apl_ids"])
apl_C = set(result_C["open_apl_ids"])

print(f"\n  APLs in BOTH models ({len(apl_B & apl_C)}):")
for sq in sorted(apl_B & apl_C):
    row = apl_cands[apl_cands["square"]==sq]
    if len(row):
        ys = row["young_share"].values[0] if "young_share" in row.columns else 0
        es = (row["Age65+"].fillna(0)/row["Population"].replace(0,1)).values[0] if "Age65+" in row.columns else 0
        print(f"    {sq}  pop={int(row['Population'].values[0])}  young={ys*100:.0f}%  elderly={es*100:.0f}%")

print(f"\n  APLs ONLY in B — uniform (replaced by something else in C) ({len(apl_B - apl_C)}):")
for sq in sorted(apl_B - apl_C):
    row = apl_cands[apl_cands["square"]==sq]
    if len(row):
        print(f"    {sq}  pop={int(row['Population'].values[0])}")

print(f"\n  NEW APLs in C — demographics shifted them here ({len(apl_C - apl_B)}):")
for sq in sorted(apl_C - apl_B):
    row = apl_cands[apl_cands["square"]==sq]
    if len(row):
        ys = row["young_share"].values[0] if "young_share" in row.columns else 0
        print(f"    {sq}  pop={int(row['Population'].values[0])}  young={ys*100:.0f}%  ← higher young share")

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
# Save CBS zone detail for Model C
rows_C = []
for i in I:
    for f in F:
        if pulp.value(result_C["x"][(i,f)]) > 0.5:
            rows_C.append({
                "cbs_square"       : cbs.iloc[i]["square"],
                "population"       : cbs.iloc[i]["Population"],
                "young_share"      : round(cbs.iloc[i]["young_share"], 3),
                "elderly_share"    : round(cbs.iloc[i]["elderly_share"], 3),
                "age_pickup_factor": round(cbs.iloc[i]["age_pickup_factor"], 3),
                "demand_index"     : round(cbs.iloc[i]["demand_index"], 1),
                "demand_shift"     : round(cbs.iloc[i]["demand_shift"], 3),
                "assigned_facility": fac_ids[f],
                "facility_type"    : fac_type[f],
                "dist_km"          : round(dist_all[i,f], 3),
                "pickup_prob"      : round(p_C[i,f], 4),
                "annual_pickups"   : round(ann_pu_C[i,f]),
                "annual_deliveries": round(ann_del_C[i,f]),
            })
            break

pd.DataFrame(rows_C).to_csv(OUTPUT_DIR + "demographic_zone_assignments.csv", index=False)

# Summary table
summary = {
    "Model": ["Baseline (35 SPs)", "Hybrid uniform (B)", "Demographic hybrid (C)"],
    "Open SPs": [35, result_B["n_sp_open"], result_C["n_sp_open"]],
    "Open APLs": [0, result_B["n_apl_open"], result_C["n_apl_open"]],
    "Effective cost €": [2_550_000, int(result_B["objective"]), int(result_C["objective"])],
    "Pop-wtd dist km": [
        round((pd.read_csv(DIST_PATH, index_col=0).reindex(cbs["square"]).fillna(999).values.min(axis=1)
               * cbs["Population"].values).sum() / cbs["Population"].sum(), 3),
        round((result_B["assign_dist"] * cbs["Population"].values).sum() / cbs["Population"].sum(), 3),
        round((result_C["assign_dist"] * cbs["Population"].values).sum() / cbs["Population"].sum(), 3),
    ],
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUTPUT_DIR + "model_comparison_summary.csv", index=False)

print("\nOutputs saved:")
print("  demographic_zone_assignments.csv  — per zone: age factors, pickup prob, demand index")
print("  model_comparison_summary.csv      — three-way comparison table")
print("\nDone.")

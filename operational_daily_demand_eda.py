"""
Operational Daily Demand EDA for eLab Business Case 2
-----------------------------------------------------

Purpose:
This script improves the initial daily demand EDA by focusing on operational demand.
It excludes Sundays from the main operational analysis, creates operational demand
baselines, analyzes peak periods, creates final KPI tables, and saves three
presentation-ready charts.

How to use:
1. Save this file in the same folder as data_Maastricht_2025.xlsx
2. Open the folder in VS Code
3. Run: python operational_daily_demand_eda.py

Outputs are saved in:
operational_daily_demand_outputs/

Expected Excel structure:
- Sheet 1: service point locations
- Sheet 2: daily activity 2025  <-- this script uses this sheet
- Sheet 3: CBS square data
- Sheet 4: road network nodes
- Sheet 5: road network edges
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. Settings
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_Maastricht_2025.xlsx"

OUTPUT_DIR = BASE_DIR / "operational_daily_demand_outputs"
PLOTS_DIR = OUTPUT_DIR / "presentation_charts"
TABLES_DIR = OUTPUT_DIR / "tables"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Second Excel sheet = daily activity data
DAILY_ACTIVITY_SHEET = 1

# Flexible weekday labels to identify Sundays
SUNDAY_LABELS = {"sun", "sunday", "zo", "zondag"}


# =============================================================================
# 2. Helper functions
# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lower_case_with_underscores."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str:
    """Find the first matching column from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise KeyError(
        f"Could not find any of these columns: {possible_names}\n"
        f"Available columns are: {list(df.columns)}"
    )


def save_plot(filename: str) -> None:
    """Save current matplotlib plot and close it."""
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def format_number(x: float) -> str:
    """Format numbers for text outputs."""
    return f"{x:,.0f}"


def is_sunday(value) -> bool:
    """Return True if a weekday value represents Sunday."""
    if pd.isna(value):
        return False
    return str(value).strip().lower() in SUNDAY_LABELS


# =============================================================================
# 3. Load and prepare data
# =============================================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}.\n"
        "Make sure this script is saved in the same folder as data_Maastricht_2025.xlsx."
    )

print("Loading daily activity data...")
activity_raw = pd.read_excel(DATA_FILE, sheet_name=DAILY_ACTIVITY_SHEET)
activity = clean_column_names(activity_raw)

print("\nColumns found in daily activity sheet:")
print(list(activity.columns))

# Robust column detection in case exact labels differ slightly
date_col = find_column(activity, ["date", "datum"])
day_index_col = find_column(activity, ["day_index", "dayindex", "day", "day_id"])
weekday_col = find_column(activity, ["day_of_week", "weekday", "week_day", "dag_van_de_week"])
location_col = find_column(activity, ["location_id", "locationid", "service_point_id", "servicepoint_id", "id"])
deliveries_col = find_column(activity, ["deliveries", "delivery", "delivered", "home_deliveries"])
pickups_col = find_column(activity, ["pickups", "pickup", "pick_ups", "picked_up"])

activity = activity[[date_col, day_index_col, weekday_col, location_col, deliveries_col, pickups_col]].copy()
activity = activity.rename(
    columns={
        date_col: "date",
        day_index_col: "day_index",
        weekday_col: "day_of_week",
        location_col: "location_id",
        deliveries_col: "deliveries",
        pickups_col: "pickups",
    }
)

activity["date"] = pd.to_datetime(activity["date"])
activity["deliveries"] = pd.to_numeric(activity["deliveries"], errors="coerce")
activity["pickups"] = pd.to_numeric(activity["pickups"], errors="coerce")

# Aggregate city-wide daily demand across all service points
daily = (
    activity
    .groupby(["date", "day_index", "day_of_week"], as_index=False)
    .agg(
        deliveries=("deliveries", "sum"),
        pickups=("pickups", "sum"),
        active_service_points=("location_id", "nunique"),
    )
)

daily["total_demand"] = daily["deliveries"] + daily["pickups"]
daily["pickup_share"] = daily["pickups"] / daily["total_demand"].replace(0, pd.NA)
daily["delivery_share"] = daily["deliveries"] / daily["total_demand"].replace(0, pd.NA)
daily["month"] = daily["date"].dt.month
daily["month_name"] = daily["date"].dt.month_name()
daily["is_sunday"] = daily["day_of_week"].apply(is_sunday)
daily["is_zero_demand_day"] = daily["total_demand"] == 0

# Main operational dataset: exclude Sundays
operational = daily[~daily["is_sunday"]].copy()

# Additional strict operational dataset: exclude Sundays and any other zero-demand days
operational_nonzero = operational[operational["total_demand"] > 0].copy()

# Save datasets
daily.to_csv(TABLES_DIR / "calendar_daily_demand_including_sundays.csv", index=False)
operational.to_csv(TABLES_DIR / "operational_daily_demand_excluding_sundays.csv", index=False)
operational_nonzero.to_csv(TABLES_DIR / "operational_daily_demand_excluding_sundays_and_zero_days.csv", index=False)


# =============================================================================
# 4. Data checks focused on operational interpretation
# =============================================================================

missing_values = activity.isna().sum().to_frame("missing_values")
missing_values["missing_share"] = missing_values["missing_values"] / len(activity)
missing_values.to_csv(TABLES_DIR / "missing_values_daily_activity.csv")

zero_non_sundays = daily[(~daily["is_sunday"]) & (daily["total_demand"] == 0)].copy()
zero_non_sundays.to_csv(TABLES_DIR / "zero_demand_non_sundays_check.csv", index=False)

print("\n================ DATA CHECKS ================")
print(f"Calendar days in dataset: {daily['date'].nunique()}")
print(f"Sundays excluded from operational analysis: {daily['is_sunday'].sum()}")
print(f"Non-Sunday zero-demand days: {len(zero_non_sundays)}")
if len(zero_non_sundays) > 0:
    print("Non-Sunday zero-demand days found. Check tables/zero_demand_non_sundays_check.csv")


# =============================================================================
# 5. Operational demand baseline
# =============================================================================

def create_baseline_table(calendar_df: pd.DataFrame, operational_df: pd.DataFrame, nonzero_df: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table for calendar vs operational demand baselines."""
    rows = []
    datasets = {
        "Calendar days incl. Sundays": calendar_df,
        "Operational days excl. Sundays": operational_df,
        "Operational non-zero days": nonzero_df,
    }

    for label, df in datasets.items():
        total = df["total_demand"].sum()
        deliveries = df["deliveries"].sum()
        pickups = df["pickups"].sum()
        rows.append({
            "baseline": label,
            "number_of_days": len(df),
            "total_parcels": total,
            "total_deliveries": deliveries,
            "total_pickups": pickups,
            "average_daily_demand": df["total_demand"].mean(),
            "median_daily_demand": df["total_demand"].median(),
            "p90_daily_demand": df["total_demand"].quantile(0.90),
            "p95_daily_demand": df["total_demand"].quantile(0.95),
            "maximum_daily_demand": df["total_demand"].max(),
            "pickup_share": pickups / total if total != 0 else pd.NA,
            "delivery_share": deliveries / total if total != 0 else pd.NA,
        })
    return pd.DataFrame(rows)


baseline_table = create_baseline_table(daily, operational, operational_nonzero)
baseline_table.to_csv(TABLES_DIR / "operational_demand_baseline_comparison.csv", index=False)

print("\n================ OPERATIONAL DEMAND BASELINE ================")
print(baseline_table)


# =============================================================================
# 6. Final EDA KPIs
# =============================================================================

annual_total = daily["total_demand"].sum()
annual_deliveries = daily["deliveries"].sum()
annual_pickups = daily["pickups"].sum()
annual_pickup_share = annual_pickups / annual_total
annual_delivery_share = annual_deliveries / annual_total

# Weekday averages based on non-Sunday operational days
weekday_avg = (
    operational
    .groupby("day_of_week")[["deliveries", "pickups", "total_demand", "pickup_share"]]
    .mean()
    .sort_values("total_demand", ascending=False)
)

highest_weekday = weekday_avg.index[0]
highest_weekday_value = weekday_avg.iloc[0]["total_demand"]

monthly_avg = daily.groupby("month")[["deliveries", "pickups", "total_demand", "pickup_share"]].mean()
monthly_total = daily.groupby("month")[["deliveries", "pickups", "total_demand"]].sum()

highest_month = monthly_avg["total_demand"].idxmax()
highest_month_value = monthly_avg.loc[highest_month, "total_demand"]

# Recommended capacity baseline: non-Sunday operational days. If zero non-Sundays exist,
# also provide the stricter non-zero operational result.
recommended_df = operational_nonzero if len(zero_non_sundays) > 0 else operational
recommended_label = "operational non-zero days" if len(zero_non_sundays) > 0 else "operational days excluding Sundays"

final_kpis = pd.DataFrame({
    "kpi": [
        "Annual total parcels",
        "Annual deliveries",
        "Annual pickups",
        "Annual pickup share",
        "Annual delivery share",
        "Calendar-day average demand",
        "Operational baseline used",
        "Number of operational days",
        "Average operational daily demand",
        "Median operational daily demand",
        "90th percentile operational demand",
        "95th percentile operational demand",
        "Maximum daily demand",
        "Highest-demand weekday",
        "Average demand on highest-demand weekday",
        "Highest-demand month",
        "Average daily demand in highest-demand month",
        "Non-Sunday zero-demand days",
    ],
    "value": [
        annual_total,
        annual_deliveries,
        annual_pickups,
        annual_pickup_share,
        annual_delivery_share,
        daily["total_demand"].mean(),
        recommended_label,
        len(recommended_df),
        recommended_df["total_demand"].mean(),
        recommended_df["total_demand"].median(),
        recommended_df["total_demand"].quantile(0.90),
        recommended_df["total_demand"].quantile(0.95),
        recommended_df["total_demand"].max(),
        highest_weekday,
        highest_weekday_value,
        highest_month,
        highest_month_value,
        len(zero_non_sundays),
    ]
})

final_kpis.to_csv(TABLES_DIR / "final_eda_kpis_operational_demand.csv", index=False)
weekday_avg.to_csv(TABLES_DIR / "weekday_average_demand_excluding_sundays.csv")
monthly_avg.to_csv(TABLES_DIR / "monthly_average_daily_demand.csv")
monthly_total.to_csv(TABLES_DIR / "monthly_total_demand.csv")

print("\n================ FINAL KPIs ================")
print(final_kpis)


# =============================================================================
# 7. Peak-period analysis
# =============================================================================

# Top demand days
top_10_days = daily.sort_values("total_demand", ascending=False).head(10).copy()
top_10_days.to_csv(TABLES_DIR / "top_10_peak_demand_days.csv", index=False)

# Compare December vs rest of year and Nov-Dec vs rest of year
december = daily[daily["month"] == 12]
non_december = daily[daily["month"] != 12]
peak_season = daily[daily["month"].isin([11, 12])]
non_peak_season = daily[~daily["month"].isin([11, 12])]

peak_period_table = pd.DataFrame({
    "period": ["December", "Rest of year", "November-December", "January-October"],
    "number_of_days": [len(december), len(non_december), len(peak_season), len(non_peak_season)],
    "average_daily_demand": [
        december["total_demand"].mean(),
        non_december["total_demand"].mean(),
        peak_season["total_demand"].mean(),
        non_peak_season["total_demand"].mean(),
    ],
    "median_daily_demand": [
        december["total_demand"].median(),
        non_december["total_demand"].median(),
        peak_season["total_demand"].median(),
        non_peak_season["total_demand"].median(),
    ],
    "p95_daily_demand": [
        december["total_demand"].quantile(0.95),
        non_december["total_demand"].quantile(0.95),
        peak_season["total_demand"].quantile(0.95),
        non_peak_season["total_demand"].quantile(0.95),
    ],
    "maximum_daily_demand": [
        december["total_demand"].max(),
        non_december["total_demand"].max(),
        peak_season["total_demand"].max(),
        non_peak_season["total_demand"].max(),
    ],
    "pickup_share": [
        december["pickups"].sum() / december["total_demand"].sum(),
        non_december["pickups"].sum() / non_december["total_demand"].sum(),
        peak_season["pickups"].sum() / peak_season["total_demand"].sum(),
        non_peak_season["pickups"].sum() / non_peak_season["total_demand"].sum(),
    ]
})

peak_period_table.to_csv(TABLES_DIR / "peak_period_analysis.csv", index=False)

print("\n================ PEAK PERIOD ANALYSIS ================")
print(peak_period_table)
print("\nTop 10 peak demand days:")
print(top_10_days[["date", "day_of_week", "deliveries", "pickups", "total_demand", "pickup_share"]])


# =============================================================================
# 8. Presentation-ready charts
# =============================================================================

print("\nCreating presentation-ready charts...")

# Chart 1: Demand over time with 7-day moving average and peak period shading
# Use calendar data here to show full annual trend, but emphasize moving average.
# For the presentation chart, exclude Sundays and all zero-demand days.
# This creates an operational view of demand instead of a raw calendar-day view.
chart_daily = daily[
    (daily["day_of_week"] != "Sun") &
    (daily["total_demand"] > 0)
].copy()
# Sort by date to ensure the moving average is calculated correctly.
chart_daily = chart_daily.sort_values("date")
# Since Post&L does not operate on Sundays, use a 6-day operating moving average.
chart_daily["total_demand_6d_ma"] = (
    chart_daily["total_demand"]
    .rolling(window=6, min_periods=1)
    .mean()
)
plt.figure(figsize=(13, 6))
# Plot daily operating demand.
plt.plot(
    chart_daily["date"],
    chart_daily["total_demand"],
    alpha=0.35,
    linewidth=1.2,
    label="Daily operating demand"
)
# Plot 6-day moving average.
plt.plot(
    chart_daily["date"],
    chart_daily["total_demand_6d_ma"],
    linewidth=2.5,
    label="6-day operating average"
)
# Highlight November-December peak period.
plt.axvspan(
    pd.Timestamp("2025-11-01"),
    pd.Timestamp("2025-12-31"),
    alpha=0.15,
    label="Nov-Dec peak period"
)
plt.xlabel("Date")
plt.ylabel("Number of parcels")
plt.title("Operating-day parcel demand peaks strongly in the year-end period")
plt.legend()
plt.grid(alpha=0.25)
save_plot("01_presentation_operational_demand_6_day_moving_average.png")

# Chart 2: Average operational demand by weekday, excluding Sundays
weekday_order_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
weekday_order_long = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Keep the original labels but try to order them if possible
available_weekdays = list(operational["day_of_week"].dropna().unique())
if set(weekday_order_short).issubset(set(available_weekdays)):
    ordered_weekdays = weekday_order_short
elif set(weekday_order_long).issubset(set(available_weekdays)):
    ordered_weekdays = weekday_order_long
else:
    ordered_weekdays = list(weekday_avg.index)

weekday_plot = (
    operational
    .groupby("day_of_week")[["deliveries", "pickups", "total_demand"]]
    .mean()
    .reindex(ordered_weekdays)
)

weekday_plot[["deliveries", "pickups"]].plot(kind="bar", stacked=True, figsize=(11, 6))
plt.xlabel("Day of week")
plt.ylabel("Average parcels per operational day")
plt.title("Tuesday creates the strongest operational demand pressure")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)
save_plot("02_presentation_operational_demand_by_weekday.png")

# Chart 3: Pickup share stability over time with average line
# Important: zero-demand days create NA pickup_share values. Matplotlib cannot plot pd.NA,
# so we convert pickup_share to numeric and remove non-operational/zero-demand days.
pickup_share_plot = daily.copy()
pickup_share_plot["pickup_share"] = pd.to_numeric(pickup_share_plot["pickup_share"], errors="coerce")
pickup_share_plot = pickup_share_plot[
    (pickup_share_plot["total_demand"] > 0) &
    (pickup_share_plot["pickup_share"].notna())
].copy()

avg_pickup_share = annual_pickup_share
plt.figure(figsize=(13, 6))
plt.plot(
    pickup_share_plot["date"],
    pickup_share_plot["pickup_share"],
    alpha=0.75,
    linewidth=1.5,
    label="Daily pickup share"
)
plt.axhline(avg_pickup_share, linestyle="--", linewidth=2, label=f"Annual average: {avg_pickup_share:.1%}")
plt.xlabel("Date")
plt.ylabel("Pickup share")
plt.title("Pickup share remains stable at around two-thirds of parcel activity")
plt.legend()
plt.grid(alpha=0.25)
save_plot("03_presentation_pickup_share_stability.png")

# =============================================================================
# 9. Ready-to-use text summary
# =============================================================================

recommended_avg = recommended_df["total_demand"].mean()
recommended_p95 = recommended_df["total_demand"].quantile(0.95)
recommended_max = recommended_df["total_demand"].max()

text_summary = f"""
Operational Daily Demand EDA - Key Findings
==========================================

1. Annual demand baseline
In 2025, Maastricht generated {format_number(annual_total)} parcel activities in total.
This consisted of {format_number(annual_deliveries)} deliveries and {format_number(annual_pickups)} pickups.
The annual pickup share was {annual_pickup_share:.2%}, while the delivery share was {annual_delivery_share:.2%}.

2. Operational demand baseline
Sundays were excluded from the main operational analysis because no regular parcel activity takes place on those days.
The recommended operational baseline is based on {recommended_label}.
Under this baseline, average daily operational demand is {recommended_avg:,.1f} parcels.
The 95th percentile operational demand is {recommended_p95:,.1f} parcels, and the maximum observed daily demand is {recommended_max:,.0f} parcels.

3. Weekday pattern
The highest-demand weekday is {highest_weekday}, with an average of {highest_weekday_value:,.1f} parcels.
This means that capacity planning should not only rely on a general average day, but should also stress-test high-demand weekdays.

4. Peak-period pattern
The highest-demand month is month {highest_month}, with an average daily demand of {highest_month_value:,.1f} parcels.
The top 10 demand days are saved in tables/top_10_peak_demand_days.csv.
These peak days are important for bounce-rate modelling because service point capacity problems are most likely to occur on high-volume days.

5. Modelling implication
Historical activity should be used to define total city-wide demand, while the pickup-versus-delivery split can start from the observed annual pickup share of {annual_pickup_share:.2%}.
Because pickup share is stable over time, distance to the nearest service point should likely be the main driver used to adjust pickup probability in the consumer behavior model.
"""

with open(TABLES_DIR / "operational_daily_demand_key_findings.txt", "w", encoding="utf-8") as f:
    f.write(text_summary)


# =============================================================================
# 10. Finish
# =============================================================================

print("\n================ DONE ================")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("\nMost important output files:")
print("- tables/operational_demand_baseline_comparison.csv")
print("- tables/final_eda_kpis_operational_demand.csv")
print("- tables/peak_period_analysis.csv")
print("- tables/top_10_peak_demand_days.csv")
print("- tables/operational_daily_demand_key_findings.txt")
print("- presentation_charts/01_presentation_demand_over_time_peak_period.png")
print("- presentation_charts/02_presentation_operational_demand_by_weekday.png")
print("- presentation_charts/03_presentation_pickup_share_stability.png")

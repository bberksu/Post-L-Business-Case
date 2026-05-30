"""
Daily Demand Patterns EDA for eLab Business Case 2
--------------------------------------------------
Author: Tristan Kluth / Team

Purpose:
This script analyzes the 2025 daily parcel activity data for Maastricht.
It focuses on city-wide daily demand patterns: deliveries, pickups,
total demand, weekday effects, monthly effects, pickup share, and peak demand.

How to use:
1. Save this file in the same folder as data_Maastricht_2025.xlsx
2. Open the folder in VS Code
3. Run: python daily_demand_eda.py

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

# The script assumes the Excel file is in the same folder as this Python file.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_Maastricht_2025.xlsx"

# Output folders for charts and tables.
OUTPUT_DIR = BASE_DIR / "eda_daily_demand_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Sheet index 1 means the second sheet in the Excel file.
DAILY_ACTIVITY_SHEET = 1


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
    """
    Find a column by checking several possible names.
    Raises a helpful error if none are found.
    """
    columns = set(df.columns)
    for name in possible_names:
        if name in columns:
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


# =============================================================================
# 3. Load data
# =============================================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}.\n"
        "Make sure this script is saved in the same folder as data_Maastricht_2025.xlsx."
    )

print("Loading daily activity data...")
activity_raw = pd.read_excel(DATA_FILE, sheet_name=DAILY_ACTIVITY_SHEET)
activity = clean_column_names(activity_raw)

print("\nColumns in daily activity sheet:")
print(list(activity.columns))
print("\nFirst rows:")
print(activity.head())


# =============================================================================
# 4. Identify relevant columns
# =============================================================================

# These lists make the script robust if the exact column names differ slightly.
date_col = find_column(activity, ["date", "datum"])
day_index_col = find_column(activity, ["day_index", "dayindex", "day", "day_id"])
weekday_col = find_column(activity, ["day_of_week", "weekday", "week_day", "dag_van_de_week"])
location_col = find_column(activity, ["location_id", "locationid", "service_point_id", "servicepoint_id", "id"])
deliveries_col = find_column(activity, ["deliveries", "delivery", "delivered", "home_deliveries"])
pickups_col = find_column(activity, ["pickups", "pickup", "pick_ups", "picked_up"])

# Keep only the columns needed for this EDA.
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

# Convert data types.
activity["date"] = pd.to_datetime(activity["date"])
activity["deliveries"] = pd.to_numeric(activity["deliveries"], errors="coerce")
activity["pickups"] = pd.to_numeric(activity["pickups"], errors="coerce")


# =============================================================================
# 5. Basic data quality checks
# =============================================================================

print("\n================ DATA QUALITY CHECKS ================")

print(f"Number of rows: {len(activity):,}")
print(f"Number of unique dates: {activity['date'].nunique()}")
print(f"Number of unique service points: {activity['location_id'].nunique()}")
print(f"Date range: {activity['date'].min().date()} to {activity['date'].max().date()}")

missing_values = activity.isna().sum().to_frame("missing_values")
missing_values["missing_share"] = missing_values["missing_values"] / len(activity)
print("\nMissing values:")
print(missing_values)
missing_values.to_csv(TABLES_DIR / "missing_values_daily_activity.csv")

# Check duplicate service point-date rows.
duplicates = activity.duplicated(subset=["date", "location_id"]).sum()
print(f"\nDuplicate date-location rows: {duplicates}")

# Check negative values.
negative_deliveries = (activity["deliveries"] < 0).sum()
negative_pickups = (activity["pickups"] < 0).sum()
print(f"Negative delivery values: {negative_deliveries}")
print(f"Negative pickup values: {negative_pickups}")

# Check completeness of all dates in 2025.
all_2025_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
missing_dates = sorted(set(all_2025_dates) - set(activity["date"].dropna().unique()))
print(f"Missing dates in 2025: {len(missing_dates)}")
if missing_dates:
    print("First missing dates:", [d.date() for d in missing_dates[:10]])

# Save potential issue rows for inspection.
issue_rows = activity[
    activity[["date", "day_index", "day_of_week", "location_id", "deliveries", "pickups"]].isna().any(axis=1)
    | (activity["deliveries"] < 0)
    | (activity["pickups"] < 0)
]
issue_rows.to_csv(TABLES_DIR / "potential_issue_rows.csv", index=False)


# =============================================================================
# 6. Aggregate city-wide daily demand
# =============================================================================

# Sum deliveries and pickups across all service points for each day.
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
daily["pickup_share"] = daily["pickups"] / daily["total_demand"]
daily["delivery_share"] = daily["deliveries"] / daily["total_demand"]
daily["month"] = daily["date"].dt.month
daily["month_name"] = daily["date"].dt.month_name()

# Save cleaned daily table.
daily.to_csv(TABLES_DIR / "daily_aggregated_demand.csv", index=False)
print("\nSaved daily aggregated demand table.")


# =============================================================================
# 7. Summary statistics
# =============================================================================

print("\n================ SUMMARY STATISTICS ================")

summary_stats = daily[["deliveries", "pickups", "total_demand", "pickup_share", "delivery_share"]].describe()
print(summary_stats)
summary_stats.to_csv(TABLES_DIR / "summary_statistics_daily_demand.csv")

# Annual totals and average shares.
annual_deliveries = daily["deliveries"].sum()
annual_pickups = daily["pickups"].sum()
annual_total = daily["total_demand"].sum()
annual_pickup_share = annual_pickups / annual_total
annual_delivery_share = annual_deliveries / annual_total

kpi_table = pd.DataFrame(
    {
        "metric": [
            "Annual deliveries",
            "Annual pickups",
            "Annual total parcels",
            "Average deliveries per day",
            "Average pickups per day",
            "Average total parcels per day",
            "Median total parcels per day",
            "90th percentile total parcels per day",
            "95th percentile total parcels per day",
            "Maximum total parcels in one day",
            "Average pickup share",
            "Average delivery share",
        ],
        "value": [
            annual_deliveries,
            annual_pickups,
            annual_total,
            daily["deliveries"].mean(),
            daily["pickups"].mean(),
            daily["total_demand"].mean(),
            daily["total_demand"].median(),
            daily["total_demand"].quantile(0.90),
            daily["total_demand"].quantile(0.95),
            daily["total_demand"].max(),
            annual_pickup_share,
            annual_delivery_share,
        ],
    }
)

print("\nKey KPI table:")
print(kpi_table)
kpi_table.to_csv(TABLES_DIR / "kpi_table_daily_demand.csv", index=False)

# Percentile table for capacity planning.
percentile_table = daily[["deliveries", "pickups", "total_demand"]].quantile(
    [0.50, 0.75, 0.90, 0.95, 0.99]
)
percentile_table.to_csv(TABLES_DIR / "demand_percentiles_capacity_planning.csv")
print("\nDemand percentiles:")
print(percentile_table)


# =============================================================================
# 8. Weekday analysis
# =============================================================================

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# If weekday names are not in English, this still works but without custom ordering.
weekday_avg = (
    daily
    .groupby("day_of_week")[["deliveries", "pickups", "total_demand", "pickup_share"]]
    .mean()
)

if set(weekday_order).issubset(set(weekday_avg.index)):
    weekday_avg = weekday_avg.reindex(weekday_order)

weekday_avg.to_csv(TABLES_DIR / "weekday_average_demand.csv")
print("\nWeekday averages:")
print(weekday_avg)


# =============================================================================
# 9. Monthly analysis
# =============================================================================

monthly_avg = daily.groupby("month")[["deliveries", "pickups", "total_demand", "pickup_share"]].mean()
monthly_total = daily.groupby("month")[["deliveries", "pickups", "total_demand"]].sum()

monthly_avg.to_csv(TABLES_DIR / "monthly_average_daily_demand.csv")
monthly_total.to_csv(TABLES_DIR / "monthly_total_demand.csv")

print("\nMonthly average daily demand:")
print(monthly_avg)


# =============================================================================
# 10. Create plots
# =============================================================================

print("\nCreating plots...")

# Plot 1: Daily total demand, deliveries and pickups over time.
plt.figure(figsize=(13, 6))
plt.plot(daily["date"], daily["total_demand"], label="Total demand")
plt.plot(daily["date"], daily["deliveries"], label="Deliveries")
plt.plot(daily["date"], daily["pickups"], label="Pickups")
plt.xlabel("Date")
plt.ylabel("Number of parcels")
plt.title("Daily Parcel Demand in Maastricht, 2025")
plt.legend()
plt.grid(alpha=0.3)
save_plot("01_daily_parcel_demand_over_time.png")

# Plot 2: Total demand only, with 7-day moving average.
daily["total_demand_7d_ma"] = daily["total_demand"].rolling(window=7, min_periods=1).mean()
plt.figure(figsize=(13, 6))
plt.plot(daily["date"], daily["total_demand"], label="Daily total demand", alpha=0.5)
plt.plot(daily["date"], daily["total_demand_7d_ma"], label="7-day moving average")
plt.xlabel("Date")
plt.ylabel("Number of parcels")
plt.title("Daily Total Demand with 7-Day Moving Average")
plt.legend()
plt.grid(alpha=0.3)
save_plot("02_total_demand_7_day_moving_average.png")

# Plot 3: Average demand by weekday.
weekday_avg[["deliveries", "pickups", "total_demand"]].plot(kind="bar", figsize=(11, 6))
plt.xlabel("Day of week")
plt.ylabel("Average number of parcels")
plt.title("Average Daily Demand by Weekday")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
save_plot("03_average_demand_by_weekday.png")

# Plot 4: Average daily demand by month.
monthly_avg[["deliveries", "pickups", "total_demand"]].plot(kind="bar", figsize=(11, 6))
plt.xlabel("Month")
plt.ylabel("Average daily parcels")
plt.title("Average Daily Demand by Month")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
save_plot("04_average_daily_demand_by_month.png")

# Plot 5: Pickup share over time.
plt.figure(figsize=(13, 6))
plt.plot(daily["date"], daily["pickup_share"])
plt.xlabel("Date")
plt.ylabel("Pickup share")
plt.title("Pickup Share Over Time")
plt.grid(alpha=0.3)
save_plot("05_pickup_share_over_time.png")

# Plot 6: Pickup share by weekday.
weekday_avg["pickup_share"].plot(kind="bar", figsize=(10, 5))
plt.xlabel("Day of week")
plt.ylabel("Average pickup share")
plt.title("Average Pickup Share by Weekday")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
save_plot("06_pickup_share_by_weekday.png")

# Plot 7: Distribution of total daily demand.
plt.figure(figsize=(9, 6))
plt.hist(daily["total_demand"], bins=30)
plt.xlabel("Total daily demand")
plt.ylabel("Number of days")
plt.title("Distribution of Daily Parcel Demand")
plt.grid(axis="y", alpha=0.3)
save_plot("07_distribution_total_daily_demand.png")

# Plot 8: Boxplot by weekday for total demand.
plt.figure(figsize=(11, 6))
box_data = []
box_labels = []
for weekday in weekday_avg.index:
    values = daily.loc[daily["day_of_week"] == weekday, "total_demand"]
    if len(values) > 0:
        box_data.append(values)
        box_labels.append(str(weekday))
plt.boxplot(box_data, labels=box_labels)
plt.xlabel("Day of week")
plt.ylabel("Total daily demand")
plt.title("Distribution of Total Demand by Weekday")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
save_plot("08_boxplot_total_demand_by_weekday.png")


# =============================================================================
# 11. Identify top demand days
# =============================================================================

top_10_days = daily.sort_values("total_demand", ascending=False).head(10)
top_10_days.to_csv(TABLES_DIR / "top_10_demand_days.csv", index=False)

print("\nTop 10 demand days:")
print(top_10_days[["date", "day_of_week", "deliveries", "pickups", "total_demand", "pickup_share"]])


# =============================================================================
# 12. Text output for presentation/report
# =============================================================================

# This creates a small text file with ready-to-use interpretation placeholders.
report_text = f"""
Daily Demand Patterns EDA - Key Findings
========================================

1. Demand baseline
In 2025, Maastricht generated {annual_total:,.0f} parcel activities in total.
This consisted of {annual_deliveries:,.0f} deliveries and {annual_pickups:,.0f} pickups.
The average daily parcel demand was {daily['total_demand'].mean():,.1f} parcels per day.

2. Pickup vs delivery split
The annual pickup share was {annual_pickup_share:.2%}, while the delivery share was {annual_delivery_share:.2%}.
This provides a first baseline for the consumer behavior model. Later, this baseline can be adjusted based on distance to the nearest service point.

3. Capacity relevance
The median daily demand was {daily['total_demand'].median():,.1f}, while the 95th percentile daily demand was {daily['total_demand'].quantile(0.95):,.1f}.
This suggests that capacity planning should not rely only on the average day. High-demand days are more relevant when trying to keep bounce rates low.

4. Modelling implication
The historical daily activity data can be used as the city-wide demand baseline for 2025. For the optimization model, demand can then be distributed across CBS squares and assigned to service points based on distance.
"""

with open(TABLES_DIR / "daily_demand_key_findings.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n================ DONE ================")
print(f"Outputs saved to: {OUTPUT_DIR}")
print("Important files:")
print("- tables/daily_aggregated_demand.csv")
print("- tables/kpi_table_daily_demand.csv")
print("- tables/demand_percentiles_capacity_planning.csv")
print("- tables/daily_demand_key_findings.txt")
print("- plots/*.png")

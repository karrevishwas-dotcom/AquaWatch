import pandas as pd
from pathlib import Path


# ==========================================
# PATHS
# ==========================================

INPUT_FILE = "data/water_quality.xlsx"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "water_quality_features.csv"


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("======================================")
print("   AQUAWATCH FEATURE ENGINEERING")
print("======================================")

print("\nLoading dataset...")

df = pd.read_excel(INPUT_FILE)

print("Original shape:", df.shape)


# ==========================================
# 2. CONVERT TIMESTAMP
# ==========================================

print("\nProcessing timestamp...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# Sort chronologically
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ==========================================
# 3. TIME FEATURES
# ==========================================

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["month"] = df["timestamp"].dt.month

print("Time features created.")


# ==========================================
# 4. USAGE HISTORY FEATURES
# ==========================================

print("\nCreating usage-history features...")

# Previous observation demand
df["demand_lag_1"] = (
    df["previous_day_water_demand_kL"]
)

# Demand from two observations earlier
df["demand_lag_2"] = (
    df["previous_day_water_demand_kL"]
    .shift(1)
)

# Demand from three observations earlier
df["demand_lag_3"] = (
    df["previous_day_water_demand_kL"]
    .shift(2)
)

# Rolling average demand
df["demand_rolling_mean_3"] = (
    df["previous_day_water_demand_kL"]
    .rolling(window=3)
    .mean()
)

print("Usage-history features created.")


# ==========================================
# 5. WEATHER FEATURES
# ==========================================

print("\nCreating weather features...")

# Temperature and humidity interaction
df["temperature_humidity"] = (
    df["temperature_C"]
    * df["humidity_pct"]
)

# Rainfall-temperature interaction
df["rainfall_temperature"] = (
    df["rainfall_mm"]
    * df["temperature_C"]
)

print("Weather features created.")


# ==========================================
# 6. SENSOR / STORAGE FEATURES
# ==========================================

print("\nCreating sensor features...")

# Storage stress indicator
df["storage_low_indicator"] = (
    df["water_storage_level_pct"] < 30
).astype(int)

# Storage-demand interaction
df["storage_demand_ratio"] = (
    df["water_storage_level_pct"]
    /
    (df["previous_day_water_demand_kL"] + 1)
)

print("Sensor features created.")


# ==========================================
# 7. NEXT-DAY DEMAND TARGET
# ==========================================

print("\nCreating next-day demand target...")

# Since the dataset is time ordered,
# the next observation's demand becomes
# the prediction target.

df["next_day_water_demand_kL"] = (
    df["previous_day_water_demand_kL"]
    .shift(-1)
)

print("Next-day demand target created.")


# ==========================================
# 8. CONTAMINATION-RISK TARGET
# ==========================================

print("\nCreating contamination-risk target...")

print(
    "NOTE: A project-defined threshold is required "
    "to convert water_quality_index into risk."
)

# Temporary threshold for dataset preparation.
# CHANGE this value if your project specification
# provides a different contamination threshold.

QUALITY_THRESHOLD = 80

df["contamination_risk"] = (
    df["water_quality_index"]
    < QUALITY_THRESHOLD
).astype(int)

print(
    "Contamination threshold:",
    QUALITY_THRESHOLD
)

print(
    "0 = Lower risk"
)

print(
    "1 = Higher risk"
)


# ==========================================
# 9. REMOVE ROWS CREATED BY LAGGING
# ==========================================

print("\nRemoving rows with missing engineered values...")

df = df.dropna().reset_index(drop=True)


# ==========================================
# 10. SAVE FEATURE-ENGINEERED DATA
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("FEATURE ENGINEERING COMPLETED")
print("======================================")

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")

for column in df.columns:
    print("-", column)

print("\nContamination risk distribution:")

print(
    df["contamination_risk"]
    .value_counts()
)

print("\nOutput file:")
print(OUTPUT_FILE)
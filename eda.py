import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# -----------------------------
# File paths
# -----------------------------

DATA_PATH = "data/water_quality.xlsx"

FIGURE_PATH = Path("reports/figures")
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_excel(DATA_PATH)

print("======================================")
print("      AQUAWATCH - EDA")
print("======================================")

print("\nDataset Shape:")
print(df.shape)


# -----------------------------
# Dataset Information
# -----------------------------

print("\nDataset Information:")
print(df.info())


# -----------------------------
# Statistical Summary
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())


# -----------------------------
# Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# Duplicate Rows
# -----------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# -----------------------------
# Target Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["water_quality_index"],
    kde=True
)

plt.xlabel("Water Quality Index")
plt.ylabel("Frequency")
plt.title("Water Quality Index Distribution")

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "water_quality_distribution.png"
)

plt.close()


# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "correlation_heatmap.png"
)

plt.close()


# -----------------------------
# Temperature vs Water Quality
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["temperature_C"],
    df["water_quality_index"]
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Water Quality Index")
plt.title("Temperature vs Water Quality")

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "temperature_vs_quality.png"
)

plt.close()


# -----------------------------
# Rainfall vs Water Quality
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["rainfall_mm"],
    df["water_quality_index"]
)

plt.xlabel("Rainfall (mm)")
plt.ylabel("Water Quality Index")
plt.title("Rainfall vs Water Quality")

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "rainfall_vs_quality.png"
)

plt.close()


# -----------------------------
# Storage Level vs Water Quality
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["water_storage_level_pct"],
    df["water_quality_index"]
)

plt.xlabel("Water Storage Level (%)")
plt.ylabel("Water Quality Index")
plt.title("Storage Level vs Water Quality")

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "storage_vs_quality.png"
)

plt.close()


print("\n======================================")
print("EDA COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated figures:")

print(
    "reports/figures/water_quality_distribution.png"
)

print(
    "reports/figures/correlation_heatmap.png"
)

print(
    "reports/figures/temperature_vs_quality.png"
)

print(
    "reports/figures/rainfall_vs_quality.png"
)

print(
    "reports/figures/storage_vs_quality.png"
)
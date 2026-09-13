import pandas as pd
import numpy as np

# Load existing dataset
file_path = "data/water_quality.xlsx"

df = pd.read_excel(file_path)

# Make results reproducible
np.random.seed(42)

# Create contamination risk
risk_score = (
    (df["water_quality_index"] < 60).astype(int)
    + (df["humidity_pct"] > 75).astype(int)
    + (df["rainfall_mm"] > 20).astype(int)
    + (df["water_storage_level_pct"] < 40).astype(int)
)

# Risk when at least 2 risk conditions occur
df["contamination_risk"] = (risk_score >= 2).astype(int)

# Save W6 dataset
output_file = "data/water_quality_w6.xlsx"

df.to_excel(output_file, index=False)

print("=" * 60)
print("W6 DATASET CREATED")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nContamination Risk Distribution:")
print(df["contamination_risk"].value_counts())

print("\n0 = Safe")
print("1 = Contamination Risk")

print("\nSaved to:")
print(output_file)
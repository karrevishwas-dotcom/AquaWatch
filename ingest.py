import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = [
    "previous_day_water_demand_kL",
    "temperature_C",
    "rainfall_mm",
    "humidity_pct",
    "peak_hour_indicator",
    "timestamp",
    "holiday_indicator",
    "population_users",
    "water_storage_level_pct",
    "zone",
    "water_quality_index"
]


def load_data():
    file_path = Path("data/water_quality.xlsx")

    df = pd.read_excel(file_path)

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates()

    print("\nFinal shape:", df.shape)

    return df


if __name__ == "__main__":
    df = load_data()
    print("\nFirst 5 records:")
    print(df.head())
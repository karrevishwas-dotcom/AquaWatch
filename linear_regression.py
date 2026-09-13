import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_excel("data/water_quality.xlsx")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. PROCESS TIMESTAMP
# ==========================================

print("\nProcessing timestamp...")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract hour from timestamp
df["hour"] = df["timestamp"].dt.hour

# Remove original timestamp
df = df.drop(columns=["timestamp"])

print("Timestamp processed successfully!")


# ==========================================
# 3. DEFINE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["water_quality_index"])

y = df["water_quality_index"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("water_quality_index")


# ==========================================
# 4. DEFINE CATEGORICAL FEATURES
# ==========================================

categorical_features = ["zone"]


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "zone_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 7. CREATE LINEAR REGRESSION MODEL
# ==========================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        LinearRegression()
    )
])


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Linear Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n======================================")
print("     LINEAR REGRESSION RESULTS")
print("======================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ==========================================
# 11. SHOW SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nSample Predictions:")
print(results.head(10))


# ==========================================
# 12. SAVE MODEL
# ==========================================

model_path = "linear_regression_water_quality.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully:")
print(model_path)

print("\n======================================")
print(" LINEAR REGRESSION COMPLETED")
print("======================================")
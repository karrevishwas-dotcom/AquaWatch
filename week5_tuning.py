import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


# ============================================================
# AQUAWATCH - WEEK 5
# MODEL TUNING + ONE STANDARD ERROR RULE
# ============================================================

DATA_PATH = "data/processed/water_quality_features.csv"
OUTPUT_DIR = "data/processed"

print("=" * 60)
print("              AQUAWATCH - WEEK 5")
print("     MODEL TUNING + ONE-STANDARD-ERROR RULE")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURE SET
# ============================================================

demand_features = [
    "previous_day_water_demand_kL",
    "temperature_C",
    "rainfall_mm",
    "humidity_pct",
    "peak_hour_indicator",
    "holiday_indicator",
    "population_users",
    "water_storage_level_pct",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_3",
    "demand_rolling_mean_3",
    "temperature_humidity",
    "rainfall_temperature",
    "storage_low_indicator",
    "storage_demand_ratio",
]

demand_target = "next_day_water_demand_kL"


# ============================================================
# 3. LINEAR REGRESSION DATA
# ============================================================

X_reg = df[demand_features]
y_reg = df[demand_target]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

print("\nRegression training samples:", len(X_train_reg))
print("Regression testing samples:", len(X_test_reg))


# ============================================================
# 4. CROSS-VALIDATION SETUP
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. REGRESSION MODELS
# ============================================================

regression_models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge alpha=0.01": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=0.01))
    ]),

    "Ridge alpha=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=0.1))
    ]),

    "Ridge alpha=1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Ridge alpha=10": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ]),

    "Lasso alpha=0.001": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=10000))
    ]),

    "Lasso alpha=0.01": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=10000))
    ]),

    "Lasso alpha=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ]),
}


# ============================================================
# 6. REGRESSION CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("REGRESSION CROSS-VALIDATION")
print("=" * 60)

regression_results = []

for name, model in regression_models.items():

    scores = cross_val_score(
        model,
        X_train_reg,
        y_train_reg,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -scores

    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    se_rmse = std_rmse / np.sqrt(len(rmse_scores))

    regression_results.append({
        "Model": name,
        "Mean_RMSE": mean_rmse,
        "Std_RMSE": std_rmse,
        "SE_RMSE": se_rmse
    })

    print(
        f"{name:25s} "
        f"Mean RMSE = {mean_rmse:.4f} | "
        f"Std = {std_rmse:.4f} | "
        f"SE = {se_rmse:.4f}"
    )


regression_results_df = pd.DataFrame(regression_results)


# ============================================================
# 7. ONE STANDARD ERROR RULE
# ============================================================

best_index = regression_results_df["Mean_RMSE"].idxmin()

best_rmse = regression_results_df.loc[
    best_index,
    "Mean_RMSE"
]

best_se = regression_results_df.loc[
    best_index,
    "SE_RMSE"
]

threshold = best_rmse + best_se


print("\n" + "=" * 60)
print("ONE-STANDARD-ERROR RULE")
print("=" * 60)

print(f"Best CV RMSE : {best_rmse:.4f}")
print(f"Best CV SE   : {best_se:.4f}")
print(f"1-SE threshold: {threshold:.4f}")


# For RMSE, lower is better.
# Select the simplest model inside the 1-SE threshold.
#
# Simplicity order:
# Linear Regression -> Ridge -> Lasso

simplicity_order = [
    "Linear Regression",
    "Ridge alpha=0.01",
    "Ridge alpha=0.1",
    "Ridge alpha=1",
    "Ridge alpha=10",
    "Lasso alpha=0.001",
    "Lasso alpha=0.01",
    "Lasso alpha=0.1",
]

eligible_models = regression_results_df[
    regression_results_df["Mean_RMSE"] <= threshold
]

selected_regression_model = None

for model_name in simplicity_order:
    if model_name in eligible_models["Model"].values:
        selected_regression_model = model_name
        break


print("Selected regression model:", selected_regression_model)


# ============================================================
# 8. TRAIN FINAL REGRESSION MODEL
# ============================================================

final_regression_model = regression_models[
    selected_regression_model
]

final_regression_model.fit(
    X_train_reg,
    y_train_reg
)

reg_predictions = final_regression_model.predict(
    X_test_reg
)


mae = mean_absolute_error(
    y_test_reg,
    reg_predictions
)

mse = mean_squared_error(
    y_test_reg,
    reg_predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test_reg,
    reg_predictions
)


print("\n" + "=" * 60)
print("FINAL REGRESSION TEST PERFORMANCE")
print("=" * 60)

print("Model:", selected_regression_model)
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - CONTAMINATION RISK")
print("=" * 60)


classification_features = [
    "temperature_C",
    "rainfall_mm",
    "humidity_pct",
    "water_storage_level_pct",
    "water_quality_index",
    "storage_low_indicator",
    "temperature_humidity",
    "rainfall_temperature",
]


classification_target = "contamination_risk"


X_cls = df[classification_features]
y_cls = df[classification_target]


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.20,
    random_state=42,
    stratify=y_cls
)


print("Classification training samples:", len(X_train_cls))
print("Classification testing samples:", len(X_test_cls))


# ============================================================
# 10. LOGISTIC CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


logistic_models = {
    "Logistic C=0.01": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.01, max_iter=5000))
    ]),

    "Logistic C=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=5000))
    ]),

    "Logistic C=1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=5000))
    ]),

    "Logistic C=10": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=10.0, max_iter=5000))
    ]),
}


classification_results = []

for name, model in logistic_models.items():

    scores = cross_val_score(
        model,
        X_train_cls,
        y_train_cls,
        cv=skf,
        scoring="f1"
    )

    mean_f1 = scores.mean()
    std_f1 = scores.std()
    se_f1 = std_f1 / np.sqrt(len(scores))

    classification_results.append({
        "Model": name,
        "Mean_F1": mean_f1,
        "Std_F1": std_f1,
        "SE_F1": se_f1
    })

    print(
        f"{name:20s} "
        f"Mean F1 = {mean_f1:.4f} | "
        f"Std = {std_f1:.4f} | "
        f"SE = {se_f1:.4f}"
    )


classification_results_df = pd.DataFrame(
    classification_results
)


# ============================================================
# 11. ONE STANDARD ERROR RULE FOR LOGISTIC MODEL
# ============================================================

best_cls_index = classification_results_df[
    "Mean_F1"
].idxmax()

best_f1 = classification_results_df.loc[
    best_cls_index,
    "Mean_F1"
]

best_f1_se = classification_results_df.loc[
    best_cls_index,
    "SE_F1"
]

f1_threshold = best_f1 - best_f1_se


print("\n" + "=" * 60)
print("LOGISTIC ONE-STANDARD-ERROR RULE")
print("=" * 60)

print(f"Best CV F1       : {best_f1:.4f}")
print(f"Best CV SE       : {best_f1_se:.4f}")
print(f"1-SE threshold   : {f1_threshold:.4f}")


# Simplicity: smaller C = stronger regularization.
classification_simplicity = [
    "Logistic C=0.01",
    "Logistic C=0.1",
    "Logistic C=1",
    "Logistic C=10",
]


eligible_cls = classification_results_df[
    classification_results_df["Mean_F1"] >= f1_threshold
]

selected_classification_model = None

for model_name in classification_simplicity:
    if model_name in eligible_cls["Model"].values:
        selected_classification_model = model_name
        break


print(
    "Selected classification model:",
    selected_classification_model
)


# ============================================================
# 12. TRAIN FINAL LOGISTIC MODEL
# ============================================================

final_classification_model = logistic_models[
    selected_classification_model
]

final_classification_model.fit(
    X_train_cls,
    y_train_cls
)

cls_predictions = final_classification_model.predict(
    X_test_cls
)


accuracy = accuracy_score(
    y_test_cls,
    cls_predictions
)

precision = precision_score(
    y_test_cls,
    cls_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_cls,
    cls_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_cls,
    cls_predictions,
    zero_division=0
)


print("\n" + "=" * 60)
print("FINAL LOGISTIC TEST PERFORMANCE")
print("=" * 60)

print("Model:", selected_classification_model)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test_cls,
        cls_predictions,
        zero_division=0
    )
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


regression_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "week5_regression_cv_results.csv"
    ),
    index=False
)


classification_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "week5_logistic_cv_results.csv"
    ),
    index=False
)


prediction_results = pd.DataFrame({
    "Actual_Next_Day_Demand_kL": y_test_reg.values,
    "Predicted_Next_Day_Demand_kL": reg_predictions
})

prediction_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "week5_final_demand_predictions.csv"
    ),
    index=False
)


classification_predictions = pd.DataFrame({
    "Actual_Contamination_Risk": y_test_cls.values,
    "Predicted_Contamination_Risk": cls_predictions
})

classification_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "week5_final_contamination_predictions.csv"
    ),
    index=False
)


# ============================================================
# 14. FINAL MILESTONE
# ============================================================

print("\n" + "=" * 60)
print("       WEEK 5 BASELINE MILESTONE COMPLETE")
print("=" * 60)

print("\nFinal demand model:")
print(selected_regression_model)

print("\nFinal contamination model:")
print(selected_classification_model)

print("\nSaved files:")
print("- data/processed/week5_regression_cv_results.csv")
print("- data/processed/week5_logistic_cv_results.csv")
print("- data/processed/week5_final_demand_predictions.csv")
print("- data/processed/week5_final_contamination_predictions.csv")

print("\nHonest baseline milestone shipped.")
print("=" * 60)
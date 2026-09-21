# ============================================================
# AquaWatch W8 - XGBoost Champion + SHAP
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "water_quality_w6.xlsx"
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

GRAPH_DIR = os.path.join(
    REPORTS_DIR,
    "xgboost_graphs"
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


print("=" * 60)
print("AquaWatch W8 - XGBoost Champion + SHAP")
print("=" * 60)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_excel(
    DATA_FILE
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. DISPLAY COLUMNS
# ============================================================

print("\nColumns:")

for column in df.columns:
    print("-", column)


# ============================================================
# 4. TARGET
# ============================================================

TARGET = "contamination_risk"

if TARGET not in df.columns:

    print(
        "\nERROR: contamination_risk column not found!"
    )

    exit()


# ============================================================
# 5. CLEAN TARGET
# ============================================================

df = df.dropna(
    subset=[TARGET]
).copy()


# If target is text, convert it

if df[TARGET].dtype == "object":

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df[TARGET] = df[TARGET].map({

        "yes": 1,
        "no": 0,

        "true": 1,
        "false": 0,

        "risk": 1,
        "no risk": 0,

        "high": 1,
        "low": 0,

        "1": 1,
        "0": 0
    })


df = df.dropna(
    subset=[TARGET]
)

df[TARGET] = df[TARGET].astype(int)


print("\nContamination Risk Distribution:")

print(
    df[TARGET].value_counts()
)


# ============================================================
# 6. SEPARATE X AND Y
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# 7. REMOVE TIMESTAMP
# ============================================================

if "timestamp" in X.columns:

    X = X.drop(
        columns=["timestamp"]
    )


# ============================================================
# 8. ENCODE CATEGORICAL FEATURES
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=False
)


# ============================================================
# 9. HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)


print(
    "\nNumber of features:",
    X.shape[1]
)


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% training
# 20% temporary data

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# Second split:
# 10% validation
# 10% test

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=42,

    stratify=y_temp
)


print("\nData split:")

print(
    "Training   :", len(X_train)
)

print(
    "Validation :", len(X_val)
)

print(
    "Testing    :", len(X_test)
)


# ============================================================
# 11. CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")


model = XGBClassifier(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=4,

    min_child_weight=2,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1,

    tree_method="hist"
)


# ============================================================
# 12. EARLY STOPPING
# ============================================================

print("\nTraining XGBoost with early stopping...")

model.fit(

    X_train,

    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=False
)


print(
    "XGBoost training completed!"
)


# ============================================================
# 13. BEST ITERATION
# ============================================================

try:

    print(
        "\nBest iteration:",
        model.best_iteration
    )

except:

    print(
        "\nBest iteration information unavailable."
    )


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. MODEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 60)
print("XGBOOST TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[0, 1]
)


tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n                 Predicted")

print(
    "                 No Risk   Risk"
)

print(
    f"Actual No Risk  {tn:8d} {fp:8d}"
)

print(
    f"Actual Risk     {fn:8d} {tp:8d}"
)

print("\nTP =", tp)
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Contamination Risk",
            "Contamination Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. XGBOOST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

)


print("\n" + "=" * 60)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.head(15).to_string(
        index=False
    )
)


# Save feature importance

importance.to_csv(

    os.path.join(
        REPORTS_DIR,
        "xgboost_feature_importance.csv"
    ),

    index=False
)


# ============================================================
# 19. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = importance.head(10)

plt.figure(
    figsize=(10, 6)
)

plt.barh(

    top_features["Feature"][::-1],

    top_features["Importance"][::-1]

)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "AquaWatch - XGBoost Feature Importance"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        GRAPH_DIR,
        "xgboost_feature_importance.png"
    ),

    dpi=300
)

plt.show()


# ============================================================
# 20. SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 60)
print("SHAP EXPLANATION")
print("=" * 60)

print(
    "\nCalculating SHAP values..."
)


explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_test
)


print(
    "SHAP values calculated successfully!"
)


# ============================================================
# 21. SHAP SUMMARY PLOT
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)


plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(

    shap_values,

    X_test,

    show=False

)

plt.title(
    "AquaWatch - SHAP Feature Impact"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        GRAPH_DIR,
        "shap_summary.png"
    ),

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ============================================================
# 22. SHAP BAR PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

shap.summary_plot(

    shap_values,

    X_test,

    plot_type="bar",

    show=False

)

plt.title(
    "AquaWatch - SHAP Global Feature Importance"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        GRAPH_DIR,
        "shap_bar.png"
    ),

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# ============================================================
# 23. EXPLAIN ONE PREDICTION
# ============================================================

sample_number = 0

sample = X_test.iloc[
    sample_number
]

sample_shap = shap_values[
    sample_number
]


# Get strongest SHAP features

shap_df = pd.DataFrame({

    "Feature": X_test.columns,

    "SHAP_Value": sample_shap,

    "Feature_Value": sample.values

})


shap_df["Absolute_SHAP"] = (
    shap_df["SHAP_Value"].abs()
)


shap_df = shap_df.sort_values(

    by="Absolute_SHAP",

    ascending=False

)


print("\n" + "=" * 60)
print("SINGLE PREDICTION EXPLANATION")
print("=" * 60)

print(
    "\nActual class:",
    y_test.iloc[sample_number]
)

print(
    "Predicted class:",
    y_pred[sample_number]
)

print(
    "Risk probability:",
    f"{y_probability[sample_number]:.4f}"
)


print(
    "\nTop factors affecting this prediction:"
)

print(

    shap_df[
        [
            "Feature",
            "SHAP_Value",
            "Feature_Value"
        ]
    ].head(10).to_string(
        index=False
    )

)


# ============================================================
# 24. SAVE SHAP RESULTS
# ============================================================

shap_df.to_csv(

    os.path.join(
        REPORTS_DIR,
        "xgboost_shap_prediction.csv"
    ),

    index=False
)


# ============================================================
# 25. SAVE RESULTS
# ============================================================

results_file = os.path.join(

    REPORTS_DIR,

    "xgboost_results.txt"

)


with open(
    results_file,
    "w"
) as f:

    f.write(
        "AquaWatch W8 - XGBoost Champion\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC   : {roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix\n"
    )

    f.write(
        f"TN = {tn}\n"
    )

    f.write(
        f"FP = {fp}\n"
    )

    f.write(
        f"FN = {fn}\n"
    )

    f.write(
        f"TP = {tp}\n\n"
    )

    f.write(
        "Top XGBoost Features\n"
    )

    f.write(
        importance.head(15).to_string(
            index=False
        )
    )


# ============================================================
# 26. FINAL
# ============================================================

print("\n" + "=" * 60)
print("W8 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nResults saved in:")
print("reports/")

print("\nGraphs saved in:")
print("reports/xgboost_graphs/")

print("\nFiles created:")

print(
    "- xgboost_results.txt"
)

print(
    "- xgboost_feature_importance.csv"
)

print(
    "- xgboost_shap_prediction.csv"
)

print(
    "- xgboost_feature_importance.png"
)

print(
    "- shap_summary.png"
)

print(
    "- shap_bar.png"
)

print("\nW8 Objective:")
print(
    "Build an XGBoost champion model, use validation "
    "to control training, and explain predictions using SHAP."
)

print(
    "\nAquaWatch priority:"
)

print(
    "Recall is important because missing a real "
    "contamination event is more serious than a false alert."
)

print("=" * 60)
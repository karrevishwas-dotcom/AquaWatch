# ============================================================
# AquaWatch W7 - Random Forest for Contamination Risk
# ============================================================

import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. FIND PROJECT FOLDER
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

print("=" * 60)
print("AquaWatch W7 - Contamination Risk Random Forest")
print("=" * 60)


# ------------------------------------------------------------
# 2. FIND DATASET
# ------------------------------------------------------------

files = []

for root, dirs, filenames in os.walk(BASE_DIR):

    for file in filenames:

        if file.endswith(".csv") or file.endswith(".xlsx"):

            files.append(
                os.path.join(root, file)
            )


dataset_path = None

for file in files:

    try:

        if file.endswith(".csv"):
            temp = pd.read_csv(file, nrows=5)

        else:
            temp = pd.read_excel(file, nrows=5)

        if "contamination_risk" in temp.columns:

            dataset_path = file
            break

    except:
        pass


if dataset_path is None:

    print("\nERROR: Dataset not found!")
    print("Dataset must contain 'contamination_risk'.")
    exit()


print("\nDataset found:")
print(dataset_path)


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

if dataset_path.endswith(".csv"):

    df = pd.read_csv(dataset_path)

else:

    df = pd.read_excel(dataset_path)


print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 4. DISPLAY COLUMNS
# ------------------------------------------------------------

print("\nColumns:")

for column in df.columns:

    print("-", column)


# ------------------------------------------------------------
# 5. TARGET
# ------------------------------------------------------------

target = "contamination_risk"

print("\nTarget:", target)


# Remove rows where target is missing

df = df.dropna(
    subset=[target]
)


# ------------------------------------------------------------
# 6. CONVERT TARGET TO 0 AND 1
# ------------------------------------------------------------

if df[target].dtype == "object":

    df[target] = (
        df[target]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df[target] = df[target].map({
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
    subset=[target]
)

df[target] = df[target].astype(int)


print("\nContamination Risk Distribution:")
print(df[target].value_counts())


# ------------------------------------------------------------
# 7. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

# IMPORTANT:
# contamination_risk is removed from X.
# This prevents target leakage.

X = df.drop(
    columns=[target]
)

y = df[target]


# ------------------------------------------------------------
# 8. REMOVE TIMESTAMP
# ------------------------------------------------------------

if "timestamp" in X.columns:

    X = X.drop(
        columns=["timestamp"]
    )


# ------------------------------------------------------------
# 9. CONVERT CATEGORICAL DATA
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    drop_first=False
)


# ------------------------------------------------------------
# 10. HANDLE MISSING VALUES
# ------------------------------------------------------------

X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)


print("\nNumber of features:", X.shape[1])


# ------------------------------------------------------------
# 11. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 12. CREATE RANDOM FOREST
# ------------------------------------------------------------

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    min_samples_split=4,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1,

    oob_score=True
)


# ------------------------------------------------------------
# 13. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully!")


# ------------------------------------------------------------
# 14. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 15. EVALUATION
# ------------------------------------------------------------

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


print("\n" + "=" * 60)
print("TEST RESULTS")
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


# ------------------------------------------------------------
# 16. OOB SCORE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OOB RESULTS")
print("=" * 60)

print(
    f"OOB Score : {model.oob_score_:.4f}"
)

print(
    f"OOB Error : {1 - model.oob_score_:.4f}"
)


# ------------------------------------------------------------
# 17. CONFUSION MATRIX
# ------------------------------------------------------------

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
print("                 No Risk   Risk")

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


# ------------------------------------------------------------
# 18. CLASSIFICATION REPORT
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 19. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

)


print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 20. SAVE RESULTS
# ------------------------------------------------------------

reports_folder = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    reports_folder,
    exist_ok=True
)


# Save feature importance

importance.to_csv(
    os.path.join(
        reports_folder,
        "random_forest_feature_importance.csv"
    ),
    index=False
)


# Save predictions

predictions = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": y_pred

})


predictions.to_csv(
    os.path.join(
        reports_folder,
        "random_forest_predictions.csv"
    ),
    index=False
)


# Save summary

with open(
    os.path.join(
        reports_folder,
        "random_forest_results.txt"
    ),
    "w"
) as f:

    f.write("AquaWatch W7 - Random Forest\n")
    f.write("============================\n\n")

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
        f"OOB Score : {model.oob_score_:.4f}\n"
    )

    f.write(
        f"OOB Error : {1 - model.oob_score_:.4f}\n\n"
    )

    f.write("Confusion Matrix\n")
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

    f.write("Top Features\n")
    f.write(
        importance.head(15).to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# 21. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("W7 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nResults saved in reports/")

print("\nFiles created:")
print("- random_forest_feature_importance.csv")
print("- random_forest_predictions.csv")
print("- random_forest_results.txt")

print("\nW7 Objective:")
print(
    "Use Random Forest to predict contamination-risk events "
    "and identify the most important factors."
)

print("\nRecall is important because AquaWatch should")
print("avoid missing real contamination events.")
print("=" * 60)

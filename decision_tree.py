# ============================================================
# AquaWatch W6 - Contamination Risk Decision Tree
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "water_quality_w6.xlsx"
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "figures"
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET = "contamination_risk"


print("=" * 60)
print("AquaWatch W6 - Contamination Risk Decision Tree")
print("=" * 60)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    print("\nERROR: Dataset file not found.")
    print("Expected location:")
    print(DATA_PATH)
    raise

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumns:")
for col in df.columns:
    print("-", col)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ============================================================
# 5. CHECK TARGET COLUMN
# ============================================================

if TARGET not in df.columns:
    print("\nERROR:")
    print(f"Target column '{TARGET}' was not found.")

    print("\nAvailable columns are:")
    for col in df.columns:
        print("-", col)

    raise ValueError(
        f"Please check that your Excel file contains '{TARGET}'."
    )


print("\nTarget:", TARGET)


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

df = df.dropna(subset=[TARGET])

# Convert target to 0/1 if necessary
if df[TARGET].dtype == "object":

    target_map = {
        "safe": 0,
        "no": 0,
        "false": 0,
        "0": 0,
        "risk": 1,
        "contamination risk": 1,
        "yes": 1,
        "true": 1,
        "1": 1
    }

    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(target_map)
    )

    df = df.dropna(subset=[TARGET])

df[TARGET] = df[TARGET].astype(int)

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\n0 = Safe")
print("1 = Contamination Risk")


# ============================================================
# 7. CONVERT TIMESTAMP
# ============================================================

print("\nPreparing features...")

if "timestamp" in df.columns:

    print("Processing timestamp...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Extract useful time information
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Remove original datetime column
    df = df.drop(columns=["timestamp"])


# ============================================================
# 8. PREPARE X AND Y
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 9. REMOVE ID COLUMNS
# ============================================================

id_columns = []

for col in X.columns:

    if (
        col.lower() == "id"
        or col.lower().endswith("_id")
        or col.lower().startswith("id_")
    ):
        id_columns.append(col)

if id_columns:
    print("\nRemoving ID columns:")
    for col in id_columns:
        print("-", col)

    X = X.drop(columns=id_columns)


# ============================================================
# 10. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumeric features:")
for col in numeric_features:
    print("-", col)

print("\nCategorical features:")
for col in categorical_features:
    print("-", col)


# ============================================================
# 11. HANDLE NUMERIC MISSING VALUES
# ============================================================

for col in numeric_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    median_value = X[col].median()

    if pd.isna(median_value):
        median_value = 0

    X[col] = X[col].fillna(median_value)


# ============================================================
# 12. HANDLE CATEGORICAL FEATURES
# ============================================================

if categorical_features:

    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=False
    )


# ============================================================
# 13. FINAL DATA CLEANING
# ============================================================

X = X.replace([float("inf"), float("-inf")], pd.NA)

X = X.fillna(0)

# Convert boolean columns to integers
for col in X.columns:

    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)


# Make sure every feature is numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(0)


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 15. TRAIN DECISION TREE
# ============================================================

print("\nTraining Decision Tree...")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Decision Tree training completed.")


# ============================================================
# 16. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 17. MODEL EVALUATION
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


print("\n")
print("=" * 60)
print("W6 MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Meaning:")
print("TN =", cm[0][0], "-> Safe correctly predicted")
print("FP =", cm[0][1], "-> Safe predicted as Risk")
print("FN =", cm[1][0], "-> Risk predicted as Safe")
print("TP =", cm[1][1], "-> Risk correctly predicted")


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Safe",
            "Contamination Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# 21. SAVE FEATURE IMPORTANCE CSV
# ============================================================

importance_csv = os.path.join(
    FIGURE_DIR,
    "contamination_feature_importance.csv"
)

feature_importance.to_csv(
    importance_csv,
    index=False
)

print("\nFeature importance saved to:")
print(importance_csv)


# ============================================================
# 22. DISPLAY TOP 5 FEATURES
# ============================================================

print("\n")
print("=" * 60)
print("TOP 5 CONTAMINATION-RISK FEATURES")
print("=" * 60)

top_features = feature_importance.head(5)

for i, row in enumerate(
    top_features.itertuples(),
    start=1
):
    print(
        f"{i}. {row.feature} "
        f"({row.importance:.4f})"
    )


# ============================================================
# 23. FIND TOP DECISION TREE SPLITS
# ============================================================

print("\n")
print("=" * 60)
print("TOP DECISION TREE SPLITS")
print("=" * 60)

tree = model.tree_

feature_names = X.columns.tolist()

split_count = 0

for node in range(tree.node_count):

    feature_index = tree.feature[node]

    if feature_index != -2:

        feature_name = feature_names[feature_index]

        threshold = tree.threshold[node]

        print(
            f"Split {split_count + 1}: "
            f"{feature_name} <= {threshold:.4f}"
        )

        split_count += 1

        if split_count >= 5:
            break


# ============================================================
# 24. PLOT DECISION TREE
# ============================================================

print("\nCreating decision tree visualization...")

plt.figure(
    figsize=(24, 14)
)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[
        "Safe",
        "Contamination Risk"
    ],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "AquaWatch W6 - Contamination Risk Decision Tree"
)

plt.tight_layout()

tree_image = os.path.join(
    FIGURE_DIR,
    "contamination_decision_tree.png"
)

plt.savefig(
    tree_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision tree saved to:")
print(tree_image)


# ============================================================
# 25. FEATURE IMPORTANCE PLOT
# ============================================================

print("\nCreating feature importance graph...")

top_plot = feature_importance.head(10)

plt.figure(
    figsize=(12, 7)
)

plt.barh(
    top_plot["feature"][::-1],
    top_plot["importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 10 Features for Contamination Risk"
)

plt.tight_layout()

importance_image = os.path.join(
    FIGURE_DIR,
    "contamination_feature_importance.png"
)

plt.savefig(
    importance_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Feature importance graph saved to:")
print(importance_image)


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("W6 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print(
    "- reports/figures/"
    "contamination_decision_tree.png"
)

print(
    "- reports/figures/"
    "contamination_feature_importance.png"
)

print(
    "- reports/figures/"
    "contamination_feature_importance.csv"
)

print("\nTop 5 features:")

for i, row in enumerate(
    top_features.itertuples(),
    start=1
):
    print(
        f"{i}. {row.feature}"
    )

print("\nW6 Decision Tree analysis completed.")
print("=" * 60)
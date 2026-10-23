
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

os.makedirs("models", exist_ok=True)

# ==================================================
# 2. LOAD DATASET
# ==================================================

dataset = load_breast_cancer(as_frame=True)

X = dataset.data.copy()
y = dataset.target.copy()

feature_names = X.columns.tolist()

# sklearn dataset mapping:
# 0 = malignant
# 1 = benign

print("\n" + "=" * 65)
print("BREAST CANCER DIAGNOSTIC - ML TRAINING")
print("=" * 65)

print("\nDataset loaded successfully!")
print(f"Dataset shape: {X.shape}")

print("\nTarget distribution:")
print(
    y.map({0: "Malignant", 1: "Benign"})
    .value_counts()
)

# ==================================================
# 3. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ==================================================
# 4. DEFINE MACHINE LEARNING MODELS
# ==================================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1
        ))
    ])
}

# ==================================================
# 5. CUSTOM MALIGNANT-CLASS SCORING FUNCTIONS
# ==================================================

# These functions are called directly by cross_validate.
# They receive the fitted estimator, validation X, and y.

def malignant_precision_scorer(estimator, X_valid, y_valid):
    predictions = estimator.predict(X_valid)

    return precision_score(
        y_valid,
        predictions,
        pos_label=0,
        zero_division=0
    )


def malignant_recall_scorer(estimator, X_valid, y_valid):
    predictions = estimator.predict(X_valid)

    return recall_score(
        y_valid,
        predictions,
        pos_label=0,
        zero_division=0
    )


def malignant_f1_scorer(estimator, X_valid, y_valid):
    predictions = estimator.predict(X_valid)

    return f1_score(
        y_valid,
        predictions,
        pos_label=0,
        zero_division=0
    )


def malignant_roc_auc_scorer(estimator, X_valid, y_valid):
    probabilities = estimator.predict_proba(X_valid)

    # Find the probability column for malignant class 0.
    malignant_index = list(estimator.classes_).index(0)
    malignant_probabilities = probabilities[:, malignant_index]

    actual_malignant = (
        np.asarray(y_valid) == 0
    ).astype(int)

    return roc_auc_score(
        actual_malignant,
        malignant_probabilities
    )


scoring = {
    "accuracy": "accuracy",
    "malignant_precision": malignant_precision_scorer,
    "malignant_recall": malignant_recall_scorer,
    "malignant_f1": malignant_f1_scorer,
    "malignant_roc_auc": malignant_roc_auc_scorer
}

# ==================================================
# 6. CROSS-VALIDATION AND MODEL TRAINING
# ==================================================

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
trained_models = {}

print("\nCROSS-VALIDATION RESULTS")
print("=" * 65)

for name, estimator in models.items():

    print(f"\nEvaluating {name}...")

    cv_results = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    # Train this model on the full training split.
    estimator.fit(X_train, y_train)

    trained_models[name] = estimator

    row = {
        "Model": name,

        "CV Accuracy": cv_results[
            "test_accuracy"
        ].mean(),

        "CV Precision": cv_results[
            "test_malignant_precision"
        ].mean(),

        "CV Recall": cv_results[
            "test_malignant_recall"
        ].mean(),

        "CV F1 Score": cv_results[
            "test_malignant_f1"
        ].mean(),

        "CV ROC AUC": cv_results[
            "test_malignant_roc_auc"
        ].mean(),

        "CV Recall Std": cv_results[
            "test_malignant_recall"
        ].std()
    }

    results.append(row)

    print(f"CV Accuracy: {row['CV Accuracy']:.4f}")
    print(f"Malignant Precision: {row['CV Precision']:.4f}")
    print(f"Malignant Recall: {row['CV Recall']:.4f}")
    print(f"Malignant F1 Score: {row['CV F1 Score']:.4f}")
    print(f"Malignant ROC-AUC: {row['CV ROC AUC']:.4f}")

# ==================================================
# 7. COMPARE AND SELECT THE BEST MODEL
# ==================================================

comparison = pd.DataFrame(results)

# Prioritize malignant recall, then malignant F1.
comparison = comparison.sort_values(
    by=["CV Recall", "CV F1 Score"],
    ascending=[False, False]
).reset_index(drop=True)

best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\n" + "=" * 65)
print("MODEL LEADERBOARD")
print("=" * 65)

print(comparison.to_string(index=False))

print(f"\nSelected model: {best_name}")

# ==================================================
# 8. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ==================================================

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

malignant_index = list(best_model.classes_).index(0)
malignant_probabilities = probabilities[:, malignant_index]

actual_malignant = (
    np.asarray(y_test) == 0
).astype(int)

test_metrics = {
    "Selected Model": best_name,

    "Test Accuracy": accuracy_score(
        y_test, predictions
    ),

    "Test Precision": precision_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    ),

    "Test Recall": recall_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    ),

    "Test F1 Score": f1_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    ),

    "Test ROC AUC": roc_auc_score(
        actual_malignant,
        malignant_probabilities
    )
}

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

for metric, value in test_metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Malignant", "Benign"],
        zero_division=0
    )
)

print("CONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print("Order = Malignant, Benign")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

# ==================================================
# 9. SAVE TRAINED MODELS AND FEATURE NAMES
# ==================================================

joblib.dump(
    best_model,
    "models/best_model.joblib"
)

joblib.dump(
    feature_names,
    "models/feature_names.joblib"
)

joblib.dump(
    trained_models,
    "models/all_models.joblib"
)

# ==================================================
# 10. SAVE MODEL COMPARISON CSV
# ==================================================

# Keep cross-validation results for every model.
# Test metrics are populated only for the selected model.

comparison["Test Accuracy"] = np.nan
comparison["Test Precision"] = np.nan
comparison["Test Recall"] = np.nan
comparison["Test F1 Score"] = np.nan
comparison["Test ROC AUC"] = np.nan

best_index = comparison.index[
    comparison["Model"] == best_name
][0]

comparison.loc[best_index, "Test Accuracy"] = (
    test_metrics["Test Accuracy"]
)

comparison.loc[best_index, "Test Precision"] = (
    test_metrics["Test Precision"]
)

comparison.loc[best_index, "Test Recall"] = (
    test_metrics["Test Recall"]
)

comparison.loc[best_index, "Test F1 Score"] = (
    test_metrics["Test F1 Score"]
)

comparison.loc[best_index, "Test ROC AUC"] = (
    test_metrics["Test ROC AUC"]
)

comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

pd.DataFrame(
    [test_metrics]
).to_csv(
    "models/test_metrics.csv",
    index=False
)

# ==================================================
# 11. TRAINING COMPLETE
# ==================================================

print("\n" + "=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nSaved files:")

print("1. models/best_model.joblib")
print("2. models/feature_names.joblib")
print("3. models/all_models.joblib")
print("4. models/model_comparison.csv")
print("5. models/test_metrics.csv")

print(
    "\nEducational project only. "
    "This model is not validated for medical diagnosis."
)

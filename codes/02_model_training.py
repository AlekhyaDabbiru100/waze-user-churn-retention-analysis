import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
VISUALS_DIR = PROJECT_DIR / "visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)


def load_cleaned_data():
    data_path = OUTPUT_DIR / "cleaned_waze_churn_data.csv"
    df_model = pd.read_csv(data_path)

    print("Loaded cleaned data:", data_path)
    print("Shape:", df_model.shape)

    return df_model


def split_data(df_model):
    X = df_model.drop(columns=["label", "churn"])
    y = df_model["churn"]

    categorical_features = ["device"]
    numeric_features = X.drop(columns=categorical_features).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train churn rate:", y_train.mean())
    print("Test churn rate:", y_test.mean())

    return X_train, X_test, y_train, y_test, categorical_features, numeric_features


def build_preprocessor(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor


def train_logistic_regression(preprocessor, X_train, X_test, y_train, y_test):
    log_reg_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]
    )

    log_reg_model.fit(X_train, y_train)

    y_pred_log = log_reg_model.predict(X_test)
    y_proba_log = log_reg_model.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression Results")
    print(classification_report(
        y_test,
        y_pred_log,
        target_names=["Retained", "Churned"]
    ))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))

    return log_reg_model, y_pred_log, y_proba_log


def train_random_forest(preprocessor, X_train, X_test, y_train, y_test):
    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    print("\nRandom Forest Results")
    print(classification_report(
        y_test,
        y_pred_rf,
        target_names=["Retained", "Churned"]
    ))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

    return rf_model, y_pred_rf, y_proba_rf


def save_model_comparison(y_test, y_pred_log, y_proba_log, y_pred_rf, y_proba_rf):
    log_report = classification_report(
        y_test,
        y_pred_log,
        output_dict=True,
        zero_division=0
    )

    rf_report = classification_report(
        y_test,
        y_pred_rf,
        output_dict=True,
        zero_division=0
    )

    model_results = pd.DataFrame({
        "model": ["Logistic Regression", "Random Forest"],
        "roc_auc": [
            roc_auc_score(y_test, y_proba_log),
            roc_auc_score(y_test, y_proba_rf)
        ],
        "accuracy": [
            log_report["accuracy"],
            rf_report["accuracy"]
        ],
        "churn_precision": [
            log_report["1"]["precision"],
            rf_report["1"]["precision"]
        ],
        "churn_recall": [
            log_report["1"]["recall"],
            rf_report["1"]["recall"]
        ],
        "churn_f1": [
            log_report["1"]["f1-score"],
            rf_report["1"]["f1-score"]
        ]
    })

    model_results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nModel Comparison:")
    print(model_results)

    return model_results


def save_confusion_matrix(y_test, y_pred_rf):
    cm = confusion_matrix(y_test, y_pred_rf)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Retained", "Churned"]
    )

    disp.plot()
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "random_forest_confusion_matrix.png")
    plt.close()


def save_roc_curve(y_test, y_proba_log, y_proba_rf):
    fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_log, tpr_log, label="Logistic Regression")
    plt.plot(fpr_rf, tpr_rf, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "roc_curve.png")
    plt.close()


def save_feature_importance(rf_model, categorical_features, numeric_features):
    encoded_cat_features = (
        rf_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )

    all_features = numeric_features + list(encoded_cat_features)

    importances = rf_model.named_steps["model"].feature_importances_

    feature_importance = pd.DataFrame({
        "feature": all_features,
        "importance": importances
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    top_features = feature_importance.head(10).sort_values("importance")

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 10 Churn Prediction Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "top_10_feature_importance.png")
    plt.close()

    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))

    return feature_importance


def save_test_predictions(X_test, y_test, y_pred_rf, y_proba_rf):
    test_predictions = X_test.copy()
    test_predictions["actual_churn"] = y_test.values
    test_predictions["rf_predicted_churn"] = y_pred_rf
    test_predictions["rf_churn_probability"] = y_proba_rf

    test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    print("\nSaved test predictions.")


def main():
    df_model = load_cleaned_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_features,
        numeric_features
    ) = split_data(df_model)

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    log_reg_model, y_pred_log, y_proba_log = train_logistic_regression(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    rf_model, y_pred_rf, y_proba_rf = train_random_forest(
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_model_comparison(
        y_test,
        y_pred_log,
        y_proba_log,
        y_pred_rf,
        y_proba_rf
    )

    save_confusion_matrix(y_test, y_pred_rf)
    save_roc_curve(y_test, y_proba_log, y_proba_rf)

    save_feature_importance(
        rf_model,
        categorical_features,
        numeric_features
    )

    save_test_predictions(X_test, y_test, y_pred_rf, y_proba_rf)

    print("\nSaved model outputs to:", OUTPUT_DIR)
    print("Saved visuals to:", VISUALS_DIR)


if __name__ == "__main__":
    main()
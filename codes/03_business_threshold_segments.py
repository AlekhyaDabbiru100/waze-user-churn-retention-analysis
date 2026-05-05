import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
VISUALS_DIR = PROJECT_DIR / "visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)


def load_test_predictions():
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    results = pd.read_csv(predictions_path)

    print("Loaded test predictions:", predictions_path)
    print("Shape:", results.shape)

    return results


def business_threshold_testing(results):
    y_test = results["actual_churn"]
    y_proba_rf = results["rf_churn_probability"]

    business_results = []

    value_saved_user = 25
    retention_offer_cost = 5

    for threshold in np.arange(0.20, 0.81, 0.05):
        preds = (y_proba_rf >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        targeted_users = tp + fp
        estimated_value = tp * value_saved_user
        campaign_cost = targeted_users * retention_offer_cost
        net_value = estimated_value - campaign_cost

        churn_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        churn_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        churn_f1 = (
            2 * churn_precision * churn_recall /
            (churn_precision + churn_recall)
            if (churn_precision + churn_recall) > 0 else 0
        )

        business_results.append({
            "threshold": round(threshold, 2),
            "true_churners_caught": tp,
            "missed_churners": fn,
            "false_alarms": fp,
            "targeted_users": targeted_users,
            "churn_precision": churn_precision,
            "churn_recall": churn_recall,
            "churn_f1": churn_f1,
            "accuracy": (preds == y_test).mean(),
            "estimated_value": estimated_value,
            "campaign_cost": campaign_cost,
            "net_value": net_value
        })

    business_df = pd.DataFrame(business_results)
    business_df = business_df.sort_values("net_value", ascending=False)

    business_df.to_csv(
        OUTPUT_DIR / "business_threshold_results.csv",
        index=False
    )

    print("\nBusiness Threshold Results:")
    print(business_df)

    return business_df


def choose_best_threshold(results, business_df):
    y_test = results["actual_churn"]
    y_proba_rf = results["rf_churn_probability"]

    best_row = business_df.iloc[0]
    chosen_threshold = best_row["threshold"]

    print("\nChosen Threshold:", chosen_threshold)
    print("Estimated Net Value:", best_row["net_value"])
    print("Churn Precision:", round(best_row["churn_precision"], 3))
    print("Churn Recall:", round(best_row["churn_recall"], 3))
    print("Targeted Users:", int(best_row["targeted_users"]))
    print("True Churners Caught:", int(best_row["true_churners_caught"]))
    print("False Alarms:", int(best_row["false_alarms"]))

    y_pred_business = (y_proba_rf >= chosen_threshold).astype(int)

    print(f"\nRandom Forest Results at Threshold {chosen_threshold}")
    print(classification_report(
        y_test,
        y_pred_business,
        target_names=["Retained", "Churned"]
    ))

    cm = confusion_matrix(y_test, y_pred_business)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Retained", "Churned"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix at Business Threshold {chosen_threshold}")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "business_threshold_confusion_matrix.png")
    plt.close()

    results["predicted_churn"] = y_pred_business

    return results, chosen_threshold


def create_risk_segments(results):
    def risk_segment(prob):
        if prob >= 0.70:
            return "High Risk"
        elif prob >= 0.40:
            return "Medium Risk"
        else:
            return "Low Risk"

    results["risk_segment"] = results["rf_churn_probability"].apply(risk_segment)

    risk_summary = results.groupby("risk_segment").agg(
        users=("actual_churn", "count"),
        actual_churn_rate=("actual_churn", "mean"),
        avg_churn_probability=("rf_churn_probability", "mean"),
        avg_sessions=("sessions", "mean"),
        avg_drives=("drives", "mean"),
        avg_activity_days=("activity_days", "mean"),
        avg_driving_days=("driving_days", "mean")
    ).sort_values("actual_churn_rate", ascending=False)

    risk_summary.to_csv(OUTPUT_DIR / "risk_segment_summary.csv")

    print("\nRisk Segment Summary:")
    print(risk_summary)

    return results, risk_summary


def create_recommendations():
    recommendations = pd.DataFrame({
        "risk_segment": ["High Risk", "Medium Risk", "Low Risk"],
        "retention_action": [
            "Send immediate personalized retention offer or app re-engagement campaign",
            "Send educational nudges, route-quality reminders, and feature prompts",
            "No discount needed; monitor with low-cost engagement messaging"
        ],
        "business_reason": [
            "Highest churn rate; worth spending retention budget",
            "Moderate churn risk; use lower-cost interventions",
            "Low churn risk; avoid wasting expensive offers"
        ]
    })

    recommendations.to_csv(
        OUTPUT_DIR / "retention_recommendations.csv",
        index=False
    )

    print("\nRetention Recommendations:")
    print(recommendations)

    return recommendations


def main():
    results = load_test_predictions()

    business_df = business_threshold_testing(results)

    results, chosen_threshold = choose_best_threshold(
        results,
        business_df
    )

    results, risk_summary = create_risk_segments(results)

    create_recommendations()

    results.to_csv(
        OUTPUT_DIR / "final_scored_test_users.csv",
        index=False
    )

    print("\nSaved business outputs to:", OUTPUT_DIR)
    print("Saved visuals to:", VISUALS_DIR)


if __name__ == "__main__":
    main()
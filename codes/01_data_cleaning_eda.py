import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "waze_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
VISUALS_DIR = PROJECT_DIR / "visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset:", DATA_PATH)
    print("Shape:", df.shape)
    return df


def clean_data(df):
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(df["label"].value_counts(dropna=False))

    df_clean = df.dropna(subset=["label"]).copy()

    df_clean["churn"] = df_clean["label"].map({
        "churned": 1,
        "retained": 0
    })

    df_clean = df_clean.drop(columns=["ID"])

    print("\nCleaned Shape:", df_clean.shape)
    print(f"Overall churn rate: {df_clean['churn'].mean():.2%}")

    return df_clean


def run_eda(df_clean):
    # Device churn
    device_churn = (
        df_clean.groupby("device")["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nChurn Rate by Device:")
    print(device_churn)

    plt.figure(figsize=(6, 4))
    device_churn.plot(kind="bar")
    plt.title("Churn Rate by Device")
    plt.ylabel("Churn Rate")
    plt.xlabel("Device")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "churn_rate_by_device.png")
    plt.close()

    # Churn by activity level
    df_eda = df_clean.copy()

    df_eda["activity_segment"] = pd.qcut(
        df_eda["activity_days"],
        q=4,
        labels=[
            "Very Low Activity",
            "Low Activity",
            "Medium Activity",
            "High Activity"
        ]
    )

    activity_churn = (
        df_eda.groupby("activity_segment", observed=False)["churn"]
        .agg(users="count", churn_rate="mean")
        .reset_index()
    )

    print("\nChurn Rate by Activity Level:")
    print(activity_churn)

    plt.figure(figsize=(8, 5))
    plt.bar(activity_churn["activity_segment"], activity_churn["churn_rate"])
    plt.title("Churn Rate by Activity Level")
    plt.ylabel("Churn Rate")
    plt.xlabel("Activity Segment")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "churn_rate_by_activity_level.png")
    plt.close()

    activity_churn.to_csv(OUTPUT_DIR / "activity_churn_summary.csv", index=False)


def feature_engineering(df_clean):
    df_model = df_clean.copy()

    df_model["km_per_drive"] = (
        df_model["driven_km_drives"] / df_model["drives"].replace(0, np.nan)
    )

    df_model["minutes_per_drive"] = (
        df_model["duration_minutes_drives"] / df_model["drives"].replace(0, np.nan)
    )

    df_model["drives_per_session"] = (
        df_model["drives"] / df_model["sessions"].replace(0, np.nan)
    )

    df_model["activity_rate"] = (
        df_model["activity_days"] /
        df_model["n_days_after_onboarding"].replace(0, np.nan)
    )

    df_model["driving_rate"] = (
        df_model["driving_days"] /
        df_model["n_days_after_onboarding"].replace(0, np.nan)
    )

    df_model["total_fav_navigations"] = (
        df_model["total_navigations_fav1"] +
        df_model["total_navigations_fav2"]
    )

    df_model = df_model.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_model


def main():
    df = load_data()
    df_clean = clean_data(df)
    run_eda(df_clean)

    df_model = feature_engineering(df_clean)

    cleaned_path = OUTPUT_DIR / "cleaned_waze_churn_data.csv"
    df_model.to_csv(cleaned_path, index=False)

    print("\nSaved cleaned data to:", cleaned_path)
    print("Saved visuals to:", VISUALS_DIR)


if __name__ == "__main__":
    main()
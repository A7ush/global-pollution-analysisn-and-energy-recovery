"""
Global Pollution Analysis & Energy Recovery
============================================
Module 2 — Classical Machine Learning
  • CO₂ Emission Forecasting    (Random Forest / XGBoost-style GBT)
  • Country Clustering           (KMeans → pollution risk profiles)
  • Anomaly Detection            (Isolation Forest — pollution spikes)
  • Renewable Potential Scoring  (Gradient Boosted Regressor)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               IsolationForest)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, silhouette_score)
from sklearn.pipeline import Pipeline

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_DIR  = Path(__file__).parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save_metrics(name: str, metrics: dict):
    df = pd.DataFrame([metrics])
    df.to_csv(MODEL_DIR / f"{name}_metrics.csv", index=False)
    print(f"  Metrics saved → models/{name}_metrics.csv")


# ── 1. CO₂ Emission Forecasting ─────────────────────────────────────────────

def train_emission_forecaster(df: pd.DataFrame):
    print("\n[1] CO₂ Emission Forecasting — Random Forest vs Gradient Boosting")
    print("-" * 58)

    FEATURES = ["gdp_per_capita_usd", "renewable_energy_pct",
                "fossil_fuel_pct", "aqi_index", "industrial_waste_mt",
                "health_cost_pct_gdp", "year", "temp_anomaly_c"]
    TARGET   = "co2_emissions_mt"

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest":       RandomForestRegressor(n_estimators=200, max_depth=12,
                                                     random_state=42, n_jobs=-1),
        "GradientBoosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                         max_depth=5, random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        cv   = cross_val_score(mdl, X, y, cv=5, scoring="r2")
        metrics = {
            "model":   name,
            "MAE":     round(mean_absolute_error(y_te, pred), 3),
            "RMSE":    round(rmse(y_te, pred), 3),
            "R2":      round(r2_score(y_te, pred), 4),
            "CV_R2":   round(cv.mean(), 4),
        }
        results[name] = {"model": mdl, "pred": pred, "metrics": metrics}
        print(f"  {name:20s}  MAE={metrics['MAE']:8.2f}  RMSE={metrics['RMSE']:8.2f}  R²={metrics['R2']:.4f}")

    # Pick the better model
    best_name = max(results, key=lambda n: results[n]["metrics"]["R2"])
    best      = results[best_name]
    save_metrics("emission_forecaster", best["metrics"])

    # Feature importance
    fi   = best["model"].feature_importances_
    fi_s = pd.Series(fi, index=FEATURES).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: actual vs predicted
    ax = axes[0]
    ax.scatter(y_te, best["pred"], alpha=0.4, s=12, color="#3498db", edgecolors="none")
    lims = [min(y_te.min(), best["pred"].min()), max(y_te.max(), best["pred"].max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Actual CO₂ (Mt)")
    ax.set_ylabel("Predicted CO₂ (Mt)")
    ax.set_title(f"Actual vs Predicted CO₂\n({best_name}, R²={best['metrics']['R2']:.3f})")
    ax.grid(linestyle="--", alpha=0.35)

    # Right: feature importance
    ax = axes[1]
    fi_s.plot.barh(ax=ax, color="#2ecc71", edgecolor="white")
    ax.set_title("Feature Importance — CO₂ Forecaster")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_emission_forecasting.png", dpi=150)
    plt.close()
    print("  Saved: 06_emission_forecasting.png")

    return best["model"]


# ── 2. Country Clustering ───────────────────────────────────────────────────

def cluster_countries(df: pd.DataFrame):
    print("\n[2] Country Clustering — Pollution Risk Profiles")
    print("-" * 50)

    FEATURES = ["co2_emissions_mt", "aqi_index", "renewable_energy_pct",
                "gdp_per_capita_usd", "health_cost_pct_gdp",
                "industrial_waste_mt"]

    country_avg = df.groupby("country")[FEATURES].mean().reset_index()
    X = country_avg[FEATURES].values

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # Elbow + silhouette to pick k
    inertias, silhouettes = [], []
    K_range = range(2, 9)
    for k in K_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, lbl))

    best_k = K_range[int(np.argmax(silhouettes))]
    print(f"  Best k={best_k}  (silhouette={max(silhouettes):.3f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels   = km_final.fit_predict(Xs)
    country_avg["cluster"] = labels

    # Profile each cluster
    profile = country_avg.groupby("cluster")[FEATURES].mean().round(2)
    profile.to_csv(MODEL_DIR / "cluster_profiles.csv")
    print("  Cluster profiles:")
    print(profile.to_string())

    # Plot: scatter in 2D (CO₂ vs AQI)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    COLORS = plt.cm.tab10.colors
    ax = axes[0]
    for cl in sorted(country_avg["cluster"].unique()):
        grp = country_avg[country_avg["cluster"] == cl]
        ax.scatter(grp["co2_emissions_mt"], grp["aqi_index"],
                   s=70, color=COLORS[cl], label=f"Cluster {cl}", alpha=0.85,
                   edgecolors="white", linewidth=0.4)
        for _, row in grp.iterrows():
            ax.annotate(row["country"], (row["co2_emissions_mt"], row["aqi_index"]),
                        fontsize=5.5, alpha=0.7)
    ax.set_xlabel("Avg CO₂ Emissions (Mt)")
    ax.set_ylabel("Avg AQI Index")
    ax.set_title(f"Country Pollution Clusters (k={best_k})")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.35)

    # Elbow curve
    ax = axes[1]
    ax.plot(list(K_range), inertias, "o-", color="#e74c3c", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(list(K_range), silhouettes, "s--", color="#3498db", linewidth=2)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia", color="#e74c3c")
    ax2.set_ylabel("Silhouette Score", color="#3498db")
    ax.set_title("Elbow Curve & Silhouette Score")
    ax.grid(linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_country_clusters.png", dpi=150)
    plt.close()
    print("  Saved: 07_country_clusters.png")

    return country_avg


# ── 3. Anomaly Detection (Pollution Spikes) ─────────────────────────────────

def detect_pollution_anomalies(df: pd.DataFrame):
    print("\n[3] Anomaly Detection — Isolation Forest (Pollution Spikes)")
    print("-" * 58)

    FEATURES = ["co2_emissions_mt", "aqi_index", "industrial_waste_mt",
                "health_cost_pct_gdp"]
    X = df[FEATURES].fillna(df[FEATURES].mean())

    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso.fit_predict(X)   # -1 = anomaly, 1 = normal

    n_anom = (df["anomaly"] == -1).sum()
    print(f"  Detected {n_anom} anomalous country-year records ({n_anom/len(df)*100:.1f}%)")
    anom_df = df[df["anomaly"] == -1][["country", "year", "co2_emissions_mt",
                                        "aqi_index"]].sort_values("aqi_index", ascending=False)
    print(anom_df.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    normal = df[df["anomaly"] == 1]
    abnorm = df[df["anomaly"] == -1]
    ax.scatter(normal["year"] + np.random.uniform(-0.3, 0.3, len(normal)),
               normal["aqi_index"], s=10, color="#3498db", alpha=0.35, label="Normal")
    ax.scatter(abnorm["year"] + np.random.uniform(-0.3, 0.3, len(abnorm)),
               abnorm["aqi_index"], s=35, color="#e74c3c", alpha=0.85,
               marker="X", label=f"Anomaly (n={n_anom})")
    ax.set_xlabel("Year")
    ax.set_ylabel("AQI Index")
    ax.set_title("Pollution Anomaly Detection — Isolation Forest")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_pollution_anomalies.png", dpi=150)
    plt.close()
    print("  Saved: 08_pollution_anomalies.png")


# ── 4. Renewable Energy Potential Scoring ───────────────────────────────────

def score_renewable_potential(df: pd.DataFrame):
    print("\n[4] Renewable Energy Potential Scoring — GBT Regressor")
    print("-" * 55)

    FEATURES = ["gdp_per_capita_usd", "co2_emissions_mt", "aqi_index",
                "health_cost_pct_gdp", "industrial_waste_mt", "year",
                "fossil_fuel_pct", "temp_anomaly_c"]
    TARGET   = "renewable_energy_pct"

    X = df[FEATURES]
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)

    gbt = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                     max_depth=4, random_state=7)
    gbt.fit(X_tr, y_tr)
    pred = gbt.predict(X_te)

    metrics = {
        "MAE":  round(mean_absolute_error(y_te, pred), 3),
        "RMSE": round(rmse(y_te, pred), 3),
        "R2":   round(r2_score(y_te, pred), 4),
    }
    print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  R²={metrics['R2']}")
    save_metrics("renewable_scorer", metrics)

    # Score latest year
    latest   = df[df["year"] == df["year"].max()].copy()
    Xl       = latest[FEATURES]
    latest["renewable_score"] = gbt.predict(Xl)
    top10    = latest.nlargest(10, "renewable_score")[["country", "renewable_energy_pct",
                                                        "renewable_score"]]
    bottom10 = latest.nsmallest(10, "renewable_score")[["country", "renewable_energy_pct",
                                                          "renewable_score"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, color in [
        (axes[0], top10,    "Top 10 Renewable Potential Countries",    "#2ecc71"),
        (axes[1], bottom10, "Bottom 10 Renewable Potential Countries", "#e74c3c"),
    ]:
        ax.barh(data["country"], data["renewable_score"], color=color, edgecolor="white")
        ax.set_xlabel("Predicted Renewable %")
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_renewable_potential.png", dpi=150)
    plt.close()
    print("  Saved: 09_renewable_potential.png")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv = DATA_DIR / "pollution_energy_data.csv"
    if not csv.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_DIR / "load_real_data.py")], check=True)

    df = pd.read_csv(csv)
    df["year"] = df["year"].astype(int)
    df = df.fillna(df.median(numeric_only=True))


    train_emission_forecaster(df)
    cluster_countries(df)
    detect_pollution_anomalies(df)
    score_renewable_potential(df)

    print("\n✓ Classical ML module complete.")

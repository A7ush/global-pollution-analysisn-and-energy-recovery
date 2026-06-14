"""
Global Pollution Analysis & Energy Recovery
============================================
Module 1 — Exploratory Data Analysis & Preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = {
    "Asia":          "#e74c3c",
    "Europe":        "#3498db",
    "North America": "#2ecc71",
    "South America": "#f39c12",
    "Africa":        "#9b59b6",
    "Middle East":   "#e67e22",
    "Oceania":       "#1abc9c",
}

def load_data(path=None):
    if path is None:
        path = Path(__file__).parent / "pollution_energy_data.csv"
    df = pd.read_csv(path, parse_dates=False)
    df["year"] = df["year"].astype(int)
    return df


def run_eda(df: pd.DataFrame):
    print("=" * 60)
    print("GLOBAL POLLUTION & ENERGY RECOVERY — EDA REPORT")
    print("=" * 60)
    print(f"\nDataset shape   : {df.shape}")
    print(f"Countries       : {df['country'].nunique()}")
    print(f"Years covered   : {df['year'].min()} – {df['year'].max()}")
    print(f"Regions         : {', '.join(df['region'].unique())}")
    print("\n--- Missing values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string() or "None")
    print("\n--- Numeric summary ---")
    print(df.describe().round(2).to_string())


def plot_global_emission_trends(df: pd.DataFrame):
    """Line chart: global CO₂ emissions by region 2000–2023."""
    region_year = (
        df.groupby(["year", "region"])["co2_emissions_mt"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for region, grp in region_year.groupby("region"):
        ax.plot(grp["year"], grp["co2_emissions_mt"],
                label=region, color=PALETTE.get(region, "#555"),
                linewidth=2.2, marker="o", markersize=3)

    ax.set_title("Global CO₂ Emissions by Region (2000–2023)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions (Mt)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_global_emission_trends.png", dpi=150)
    plt.close()
    print("Saved: 01_global_emission_trends.png")


def plot_renewable_adoption(df: pd.DataFrame):
    """Area-fill chart: average renewable % per region over time."""
    region_year = (
        df.groupby(["year", "region"])["renewable_energy_pct"]
        .mean()
        .unstack("region")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    region_year.plot.area(ax=ax, colormap="tab10", alpha=0.75)
    ax.set_title("Renewable Energy Adoption by Region (% avg)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Renewable Energy (%)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_renewable_adoption.png", dpi=150)
    plt.close()
    print("Saved: 02_renewable_adoption.png")


def plot_emission_vs_gdp(df: pd.DataFrame):
    """Scatter: GDP per capita vs CO₂ for 2023 with region colour."""
    latest = df[df["year"] == df["year"].max()].copy()
    fig, ax = plt.subplots(figsize=(11, 7))

    for region, grp in latest.groupby("region"):
        ax.scatter(grp["gdp_per_capita_usd"], grp["co2_emissions_mt"],
                   label=region, color=PALETTE.get(region, "#555"),
                   s=grp["renewable_energy_pct"] * 4 + 30, alpha=0.8, edgecolors="white", linewidth=0.4)

    ax.set_title("GDP per Capita vs CO₂ Emissions (2023)\n(bubble size ∝ renewable energy %)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("GDP per Capita (USD)")
    ax.set_ylabel("CO₂ Emissions (Mt)")
#   ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_emission_vs_gdp.png", dpi=150)
    plt.close()
    print("Saved: 03_emission_vs_gdp.png")


def plot_aqi_heatmap(df: pd.DataFrame):
    """Heatmap: top-20 most-polluted countries, AQI over years."""
    top20 = (df.groupby("country")["aqi_index"].mean()
               .nlargest(20).index.tolist())
    pivot = df[df["country"].isin(top20)].pivot_table(
        index="country", columns="year", values="aqi_index", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Air Quality Index Heatmap — Top 20 Countries (2000–2023)",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="AQI Index")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_aqi_heatmap.png", dpi=150)
    plt.close()
    print("Saved: 04_aqi_heatmap.png")


def plot_energy_mix_2023(df: pd.DataFrame):
    """Grouped bar: energy mix breakdown for 10 major economies in 2023."""
    top10 = ["China", "USA", "India", "Germany", "UK",
             "Brazil", "Norway", "Saudi Arabia", "Japan", "Australia"]
    latest = df[(df["year"] == df["year"].max()) & (df["country"].isin(top10))].set_index("country")

    cols  = ["fossil_fuel_pct", "solar_energy_pct", "wind_energy_pct",
             "hydro_energy_pct", "nuclear_energy_pct"]
    labels = ["Fossil Fuel", "Solar", "Wind", "Hydro", "Nuclear"]
    colors = ["#e74c3c", "#f1c40f", "#3498db", "#27ae60", "#8e44ad"]

    mix = latest[cols].reindex(top10)
    x   = np.arange(len(top10))
    w   = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (col, lbl, clr) in enumerate(zip(cols, labels, colors)):
        ax.bar(x + i * w, mix[col], w, label=lbl, color=clr, alpha=0.88)

    ax.set_title("Energy Mix for Major Economies (2023)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(top10, rotation=30, ha="right")
    ax.set_ylabel("Share (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_energy_mix_2023.png", dpi=150)
    plt.close()
    print("Saved: 05_energy_mix_2023.png")


def preprocess(df: pd.DataFrame):
    """Return clean feature matrix + target series for ML modules."""
    features = [
        "co2_emissions_mt",
        "gdp_per_capita_usd",
        "renewable_energy_pct",
        "fossil_fuel_pct",
        "solar_energy_pct",
        "wind_energy_pct",
        "hydro_energy_pct",
        "nuclear_energy_pct",
        "aqi_index",
        "temp_anomaly_c",
        "industrial_waste_mt",
        "health_cost_pct_gdp",
        "year",
    ]
    df_clean = df[features].dropna().copy()
    return df_clean


if __name__ == "__main__":
    # Generate fresh data if CSV doesn't exist
    csv_path = Path(__file__).parent / "data" / "pollution_energy_data.csv"
    if not csv_path.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).parent / "data" / "load_real_data.py")], check=True)

    df = load_data(csv_path)
    run_eda(df)
    plot_global_emission_trends(df)
    plot_renewable_adoption(df)
    plot_emission_vs_gdp(df)
    plot_aqi_heatmap(df)
    plot_energy_mix_2023(df)
    print("\nEDA complete — all plots saved to outputs/")

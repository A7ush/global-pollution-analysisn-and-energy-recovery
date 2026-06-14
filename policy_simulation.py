"""
Global Pollution Analysis & Energy Recovery
============================================
Module 4 — Policy Impact Simulation & Energy Recovery Scoring

Real-world use cases:
  • Simulate the CO₂ impact of switching fossil → renewables
  • Policy scenario comparison (Paris Agreement targets)
  • Energy recovery potential index per country
  • Health-cost savings projection from pollution reduction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load(csv=None):
    if csv is None:
        csv = DATA_DIR / "pollution_energy_data.csv"
    df = pd.read_csv(csv)
    df["year"] = df["year"].astype(int)
    return df


# ── 1. Policy Scenario Simulation ────────────────────────────────────────────

SCENARIOS = {
    "Business as Usual":    {"renew_boost": 0.0,  "fossil_cut": 0.0},
    "Moderate Transition":  {"renew_boost": 0.15, "fossil_cut": 0.20},
    "Paris Agreement":      {"renew_boost": 0.30, "fossil_cut": 0.45},
    "Aggressive Green":     {"renew_boost": 0.55, "fossil_cut": 0.70},
}

HIGHLIGHT = {
    "China": "#e74c3c", "USA": "#3498db", "India": "#f39c12",
    "Germany": "#2ecc71", "Brazil": "#9b59b6", "Saudi Arabia": "#e67e22",
}

def simulate_policy_scenarios(df: pd.DataFrame):
    print("\n[4-1] Policy Scenario Simulation")
    print("-" * 40)

    latest = df[df["year"] == df["year"].max()].copy()
    # Simple physics-based model: emission ∝ fossil share, inverse ∝ renewable share
    results = []
    for s_name, params in SCENARIOS.items():
        row_list = []
        for _, row in latest.iterrows():
            new_renew  = min(row["renewable_energy_pct"] * (1 + params["renew_boost"]), 95)
            new_fossil = max(row["fossil_fuel_pct"]     * (1 - params["fossil_cut"]),  2)
            renew_ratio  = new_renew  / max(row["renewable_energy_pct"], 0.1)
            fossil_ratio = new_fossil / max(row["fossil_fuel_pct"],      0.1)
            new_emission = row["co2_emissions_mt"] * (fossil_ratio * 0.7 + (2 - renew_ratio) * 0.3)
            new_aqi      = row["aqi_index"] * (fossil_ratio * 0.6 + 0.4)
            new_health   = row["health_cost_pct_gdp"] * (new_aqi / row["aqi_index"])
            row_list.append({
                "country":        row["country"],
                "region":         row["region"],
                "scenario":       s_name,
                "co2_emissions":  round(new_emission, 2),
                "aqi":            round(new_aqi, 1),
                "health_cost":    round(new_health, 4),
            })
        results.extend(row_list)

    sim_df = pd.DataFrame(results)

    # Global totals per scenario
    totals = sim_df.groupby("scenario")[["co2_emissions"]].sum().reindex(SCENARIOS.keys())
    base   = totals.loc["Business as Usual", "co2_emissions"]
    totals["pct_reduction"] = ((base - totals["co2_emissions"]) / base * 100).round(1)
    print(totals.to_string())

    # Plot 1 — global CO₂ bar per scenario
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    ax = axes[0]
    bars = ax.bar(totals.index, totals["co2_emissions"], color=colors, edgecolor="white", width=0.5)
    ax.set_title("Global CO₂ Under Policy Scenarios (2023 baseline)")
    ax.set_ylabel("Total CO₂ Emissions (Mt)")
    ax.set_xticklabels(totals.index, rotation=15, ha="right")
    for bar, pct in zip(bars, totals["pct_reduction"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"−{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Plot 2 — top emitters reduction waterfall for Paris scenario
    paris = sim_df[sim_df["scenario"] == "Paris Agreement"].set_index("country")
    bau   = sim_df[sim_df["scenario"] == "Business as Usual"].set_index("country")
    top10 = bau["co2_emissions"].nlargest(10).index
    savings = (bau.loc[top10, "co2_emissions"] - paris.loc[top10, "co2_emissions"]).sort_values()

    ax = axes[1]
    colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in savings]
    ax.barh(savings.index, savings.values, color=colors_bar, edgecolor="white")
    ax.set_title("CO₂ Savings vs BAU (Paris Agreement) — Top 10 Emitters")
    ax.set_xlabel("CO₂ Savings (Mt)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_policy_scenarios.png", dpi=150)
    plt.close()
    print("  Saved: 13_policy_scenarios.png")

    return sim_df


# ── 2. Energy Recovery Potential Index ───────────────────────────────────────

def energy_recovery_index(df: pd.DataFrame):
    """
    Composite scoring (0–100) based on:
      • Waste-to-energy potential (industrial waste × 40%)
      • Fossil-to-renewable conversion headroom (× 35%)
      • Economic capability (GDP normalised × 25%)
    Real-world use: prioritise international climate-finance allocation.
    """
    print("\n[4-2] Energy Recovery Potential Index")
    print("-" * 40)

    latest = df[df["year"] == df["year"].max()].copy()

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    latest["waste_score"]    = norm(latest["industrial_waste_mt"])
    latest["headroom_score"] = norm(100 - latest["renewable_energy_pct"])
    latest["gdp_score"]      = norm(latest["gdp_per_capita_usd"])

    latest["recovery_index"] = (
        latest["waste_score"]    * 0.40 +
        latest["headroom_score"] * 0.35 +
        latest["gdp_score"]      * 0.25
    ) * 100

    ranked = latest.sort_values("recovery_index", ascending=False)[
        ["country", "region", "co2_emissions_mt", "renewable_energy_pct",
         "recovery_index"]].reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked.to_csv(OUTPUT_DIR / "energy_recovery_index.csv", index=False)
    print(ranked.head(15).to_string(index=False))

    top15 = ranked.head(15)
    colors = [{"Asia": "#e74c3c", "Europe": "#3498db", "North America": "#2ecc71",
               "South America": "#f39c12", "Africa": "#9b59b6",
               "Middle East": "#e67e22", "Oceania": "#1abc9c"}.get(r, "#555")
              for r in top15["region"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top15["country"][::-1], top15["recovery_index"][::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Energy Recovery Potential Index (0–100)")
    ax.set_title("Top 15 Countries — Energy Recovery Potential Index")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    handles = [mpatches.Patch(color=c, label=r)
               for r, c in {"Asia": "#e74c3c", "Europe": "#3498db",
                              "North America": "#2ecc71", "Africa": "#9b59b6",
                              "Middle East": "#e67e22", "South America": "#f39c12"}.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_energy_recovery_index.png", dpi=150)
    plt.close()
    print("  Saved: 14_energy_recovery_index.png")

    return ranked


# ── 3. Health-Cost Savings Projection ────────────────────────────────────────

def health_savings_projection(df: pd.DataFrame):
    """
    Polynomial regression: project health-cost % GDP savings if AQI improves
    by 10 / 20 / 30 % by 2030.  Directly quantifiable for policy makers.
    """
    print("\n[4-3] Health-Cost Savings Projection (2024–2030)")
    print("-" * 50)

    years    = np.arange(2000, 2024)
    avg_hc   = df.groupby("year")["health_cost_pct_gdp"].mean().values
    avg_aqi  = df.groupby("year")["aqi_index"].mean().values

    # Polynomial regression for trend extrapolation
    poly = PolynomialFeatures(degree=2)
    X_tr = poly.fit_transform(years.reshape(-1, 1))
    reg  = LinearRegression().fit(X_tr, avg_hc)

    future_years = np.arange(2024, 2031)
    X_fut        = poly.transform(future_years.reshape(-1, 1))
    baseline     = reg.predict(X_fut)

    projections = {
        "Baseline (BAU)":   baseline,
        "AQI −10%":         baseline * 0.90,
        "AQI −20%":         baseline * 0.78,
        "AQI −30%":         baseline * 0.64,
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, avg_hc, "o-", color="#555", linewidth=2, label="Historical avg")

    styles = [("--", "#e74c3c"), ("-.", "#f39c12"), (":", "#2ecc71"), ("-", "#3498db")]
    for (label, vals), (ls, clr) in zip(projections.items(), styles):
        ax.plot(future_years, vals, ls, color=clr, linewidth=2, label=label)

    ax.axvline(2023.5, color="gray", linestyle=":", linewidth=1)
    ax.text(2023.7, ax.get_ylim()[1] * 0.97, "Projection →", fontsize=9, color="gray")
    ax.set_title("Health-Cost Savings Projection Under AQI Reduction Scenarios (2024–2030)")
    ax.set_xlabel("Year");  ax.set_ylabel("Health Cost (% of GDP)")
    ax.legend(fontsize=9);  ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "15_health_savings_projection.png", dpi=150)
    plt.close()
    print("  Saved: 15_health_savings_projection.png")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load()
    simulate_policy_scenarios(df)
    energy_recovery_index(df)
    health_savings_projection(df)
    print("\n✓ Policy & Energy Recovery module complete.")

"""
Global Pollution Analysis & Energy Recovery
============================================
Dataset Generator — creates realistic synthetic global pollution + energy data
covering 50 countries, 2000–2023, with emissions, renewable adoption, AQI,
and economic indicators.

fake dataset 
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

COUNTRIES = {
    "China":         {"base_emission": 9000, "gdp_base": 10000, "renew_base": 0.12, "region": "Asia"},
    "USA":           {"base_emission": 5000, "gdp_base": 55000, "renew_base": 0.14, "region": "North America"},
    "India":         {"base_emission": 2600, "gdp_base": 2000,  "renew_base": 0.08, "region": "Asia"},
    "Russia":        {"base_emission": 1700, "gdp_base": 12000, "renew_base": 0.18, "region": "Europe"},
    "Japan":         {"base_emission": 1100, "gdp_base": 40000, "renew_base": 0.20, "region": "Asia"},
    "Germany":       {"base_emission": 750,  "gdp_base": 45000, "renew_base": 0.35, "region": "Europe"},
    "South Korea":   {"base_emission": 600,  "gdp_base": 30000, "renew_base": 0.08, "region": "Asia"},
    "Canada":        {"base_emission": 560,  "gdp_base": 48000, "renew_base": 0.65, "region": "North America"},
    "Brazil":        {"base_emission": 450,  "gdp_base": 10000, "renew_base": 0.45, "region": "South America"},
    "Indonesia":     {"base_emission": 600,  "gdp_base": 4000,  "renew_base": 0.12, "region": "Asia"},
    "UK":            {"base_emission": 380,  "gdp_base": 42000, "renew_base": 0.38, "region": "Europe"},
    "France":        {"base_emission": 310,  "gdp_base": 43000, "renew_base": 0.22, "region": "Europe"},
    "Australia":     {"base_emission": 390,  "gdp_base": 52000, "renew_base": 0.30, "region": "Oceania"},
    "Mexico":        {"base_emission": 450,  "gdp_base": 10000, "renew_base": 0.23, "region": "North America"},
    "Italy":         {"base_emission": 330,  "gdp_base": 33000, "renew_base": 0.36, "region": "Europe"},
    "Saudi Arabia":  {"base_emission": 620,  "gdp_base": 22000, "renew_base": 0.02, "region": "Middle East"},
    "Turkey":        {"base_emission": 440,  "gdp_base": 12000, "renew_base": 0.28, "region": "Europe"},
    "Poland":        {"base_emission": 340,  "gdp_base": 18000, "renew_base": 0.14, "region": "Europe"},
    "Spain":         {"base_emission": 260,  "gdp_base": 31000, "renew_base": 0.44, "region": "Europe"},
    "Argentina":     {"base_emission": 195,  "gdp_base": 11000, "renew_base": 0.32, "region": "South America"},
    "Thailand":      {"base_emission": 280,  "gdp_base": 7500,  "renew_base": 0.15, "region": "Asia"},
    "Nigeria":       {"base_emission": 110,  "gdp_base": 2200,  "renew_base": 0.20, "region": "Africa"},
    "Egypt":         {"base_emission": 200,  "gdp_base": 3600,  "renew_base": 0.10, "region": "Africa"},
    "Pakistan":      {"base_emission": 185,  "gdp_base": 1500,  "renew_base": 0.30, "region": "Asia"},
    "Bangladesh":    {"base_emission": 95,   "gdp_base": 1900,  "renew_base": 0.05, "region": "Asia"},
    "Vietnam":       {"base_emission": 230,  "gdp_base": 3800,  "renew_base": 0.13, "region": "Asia"},
    "Netherlands":   {"base_emission": 150,  "gdp_base": 53000, "renew_base": 0.28, "region": "Europe"},
    "Belgium":       {"base_emission": 100,  "gdp_base": 47000, "renew_base": 0.22, "region": "Europe"},
    "Sweden":        {"base_emission": 45,   "gdp_base": 55000, "renew_base": 0.65, "region": "Europe"},
    "Norway":        {"base_emission": 40,   "gdp_base": 82000, "renew_base": 0.72, "region": "Europe"},
    "Denmark":       {"base_emission": 32,   "gdp_base": 62000, "renew_base": 0.60, "region": "Europe"},
    "Switzerland":   {"base_emission": 38,   "gdp_base": 80000, "renew_base": 0.61, "region": "Europe"},
    "Austria":       {"base_emission": 65,   "gdp_base": 50000, "renew_base": 0.72, "region": "Europe"},
    "Portugal":      {"base_emission": 48,   "gdp_base": 23000, "renew_base": 0.60, "region": "Europe"},
    "Finland":       {"base_emission": 42,   "gdp_base": 49000, "renew_base": 0.42, "region": "Europe"},
    "Chile":         {"base_emission": 89,   "gdp_base": 15000, "renew_base": 0.47, "region": "South America"},
    "Colombia":      {"base_emission": 95,   "gdp_base": 6500,  "renew_base": 0.65, "region": "South America"},
    "UAE":           {"base_emission": 200,  "gdp_base": 44000, "renew_base": 0.04, "region": "Middle East"},
    "Morocco":       {"base_emission": 65,   "gdp_base": 3700,  "renew_base": 0.35, "region": "Africa"},
    "Ethiopia":      {"base_emission": 20,   "gdp_base": 950,   "renew_base": 0.90, "region": "Africa"},
    "Kenya":         {"base_emission": 18,   "gdp_base": 2000,  "renew_base": 0.75, "region": "Africa"},
    "Ghana":         {"base_emission": 22,   "gdp_base": 2500,  "renew_base": 0.40, "region": "Africa"},
    "Philippines":   {"base_emission": 130,  "gdp_base": 3500,  "renew_base": 0.28, "region": "Asia"},
    "Malaysia":      {"base_emission": 230,  "gdp_base": 12000, "renew_base": 0.18, "region": "Asia"},
    "Singapore":     {"base_emission": 52,   "gdp_base": 65000, "renew_base": 0.04, "region": "Asia"},
    "New Zealand":   {"base_emission": 32,   "gdp_base": 44000, "renew_base": 0.82, "region": "Oceania"},
    "Ireland":       {"base_emission": 37,   "gdp_base": 77000, "renew_base": 0.42, "region": "Europe"},
    "Czech Republic":{"base_emission": 100,  "gdp_base": 25000, "renew_base": 0.16, "region": "Europe"},
    "Romania":       {"base_emission": 70,   "gdp_base": 14000, "renew_base": 0.44, "region": "Europe"},
    "Ukraine":       {"base_emission": 170,  "gdp_base": 3500,  "renew_base": 0.12, "region": "Europe"},
}

YEARS = list(range(2000, 2024))

def generate_pollution_data():
    rows = []
    for country, meta in COUNTRIES.items():
        gdp = meta["gdp_base"]
        renew = meta["renew_base"]
        emission = meta["base_emission"]

        for year in YEARS:
            t = year - 2000

            # GDP grows with noise
            gdp_growth = 1 + np.random.normal(0.025, 0.012)
            gdp *= gdp_growth

            # Renewable energy adoption accelerates post-2010
            if year >= 2010:
                renew = min(renew * (1 + np.random.normal(0.045, 0.015)), 0.95)
            else:
                renew = min(renew * (1 + np.random.normal(0.015, 0.010)), 0.95)

            # Emissions: grow with GDP but curbed by renewables
            emission_change = (gdp_growth - 1) * 0.6 - (renew - meta["renew_base"]) * 2.5
            emission *= (1 + emission_change + np.random.normal(0, 0.02))
            emission = max(emission, 1)

            # AQI: correlated with emission intensity
            emission_per_gdp = emission / (gdp * 1e-3)
            aqi = 50 + emission_per_gdp * 12 + np.random.normal(0, 15)
            aqi = np.clip(aqi, 10, 500)

            # Energy mix
            fossil_pct   = max(0.05, (1 - renew) * (0.85 + np.random.normal(0, 0.04)))
            solar_pct    = renew * np.random.uniform(0.15, 0.45)
            wind_pct     = renew * np.random.uniform(0.15, 0.45)
            hydro_pct    = max(0, renew - solar_pct - wind_pct) * np.random.uniform(0.4, 0.9)
            nuclear_pct  = max(0, 1 - fossil_pct - solar_pct - wind_pct - hydro_pct)

            # Temperature anomaly proxy (rising globally)
            temp_anomaly = 0.1 * t / 10 + np.random.normal(0, 0.15)

            # Industrial waste (tons per capita, declining w/ tech)
            waste = (emission / 1000) * (1.5 - 0.02 * t) + np.random.normal(0, 0.2)
            waste = max(0.01, waste)

            # Pollution health cost (% of GDP)
            health_cost_pct = 0.005 + (aqi / 500) * 0.08 + np.random.normal(0, 0.005)
            health_cost_pct = np.clip(health_cost_pct, 0.001, 0.15)

            rows.append({
                "country":               country,
                "region":                meta["region"],
                "year":                  year,
                "co2_emissions_mt":      round(emission, 2),
                "gdp_per_capita_usd":    round(gdp, 2),
                "renewable_energy_pct":  round(renew * 100, 2),
                "fossil_fuel_pct":       round(fossil_pct * 100, 2),
                "solar_energy_pct":      round(solar_pct * 100, 2),
                "wind_energy_pct":       round(wind_pct * 100, 2),
                "hydro_energy_pct":      round(hydro_pct * 100, 2),
                "nuclear_energy_pct":    round(nuclear_pct * 100, 2),
                "aqi_index":             round(aqi, 1),
                "temp_anomaly_c":        round(temp_anomaly, 3),
                "industrial_waste_mt":   round(waste, 3),
                "health_cost_pct_gdp":   round(health_cost_pct * 100, 3),
                "population_millions":   round(np.random.uniform(0.5, 1400), 1),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    out = Path(__file__).parent
    df = generate_pollution_data()
    df.to_csv(out / "pollution_energy_data.csv", index=False)
    print(f"Dataset saved: {len(df)} rows × {df.shape[1]} columns")
    print(df.head())

# 🌍 Global Pollution Analysis & Energy Recovery

> **A full-stack Machine Learning project analysing CO₂ emissions, air quality,
> and renewable energy adoption across 50 countries (2000–2023), with deep-learning
> forecasting and real-world policy simulation.**

---

## 🏆 Resume Summary

| Dimension | Detail |
|-----------|--------|
| **Stack** | Python · Pandas · NumPy · Matplotlib · Scikit-learn · TensorFlow/Keras |
| **ML types** | Supervised (Regression) · Unsupervised (Clustering, Anomaly Detection) · Deep Learning |
| **DL architectures** | LSTM · Deep Neural Network · Autoencoder |
| **Dataset** | 50 countries × 24 years = 1 200 rows, 17 engineered features |
| **Real-world domain** | Climate policy · Environmental economics · Public health |
| **Outputs** | 15 publication-quality plots · Trained models · Policy report CSV |

---

## 🗂 Project Structure

```
global_pollution_energy_recovery/
├── data/
│   ├── generate_dataset.py      # Synthetic global dataset generator
│   └── pollution_energy_data.csv
├── eda_preprocessing.py         # EDA + 5 visualisation plots
├── classical_ml.py              # RF · GBT · KMeans · Isolation Forest
├── deep_learning.py             # LSTM · DNN · Autoencoder
├── policy_simulation.py         # Scenario sim · Recovery index · Health savings
├── run_pipeline.py              # Master runner (end-to-end)
├── requirements.txt
├── outputs/                     # All generated plots + CSV reports
└── models/                      # Saved model artifacts + metric logs
```

---

## ⚙️ Quick Start

```bash
# 1. Clone / download the project
cd global_pollution_energy_recovery

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run full pipeline (requires TensorFlow)
python run_pipeline.py

# 3b. Skip deep learning (no TensorFlow needed)
python run_pipeline.py --skip-dl

# 4. Or run modules individually
python data/generate_dataset.py
python eda_preprocessing.py
python classical_ml.py
python deep_learning.py
python policy_simulation.py
```

---

## 📊 Modules & Techniques

### Module 1 — EDA & Preprocessing (`eda_preprocessing.py`)

| Plot | Insight |
|------|---------|
| Global CO₂ by region (line) | Asia dominates and is diverging from Europe |
| Renewable adoption area chart | Nordic/South American acceleration post-2010 |
| GDP vs CO₂ scatter (bubble) | Wealthier nations are decoupling growth from emissions |
| AQI heatmap (top 20) | Middle East & South/East Asia consistently exceed WHO limits |
| Energy mix grouped bar | Germany's coal-exit vs Saudi Arabia's fossil lock-in |

**Preprocessing** produces a clean feature matrix for all downstream ML modules.

---

### Module 2 — Classical ML (`classical_ml.py`)

#### 2a. CO₂ Emission Forecasting
- **Random Forest** vs **Gradient Boosting Regressor**
- 5-fold cross-validation; best model selected by R²
- Feature importance chart → `year`, `fossil_fuel_pct`, `gdp_per_capita` dominate

#### 2b. Country Clustering
- **KMeans** with elbow + silhouette selection
- Segments countries into pollution risk profiles:
  - *High-emission / low-renewable* (China, India, Saudi Arabia)
  - *Transitioning* (USA, Germany, Australia)
  - *Green leaders* (Norway, Sweden, New Zealand)

#### 2c. Anomaly Detection
- **Isolation Forest** (5% contamination)
- Detects unexpected CO₂/AQI spikes (e.g. industrial accidents, energy crises)

#### 2d. Renewable Potential Scoring
- **GBT Regressor** predicts achievable renewable % from economic & pollution features
- Ranks countries by untapped green energy potential

---

### Module 3 — Deep Learning (`deep_learning.py`)

#### 3a. LSTM CO₂ Forecaster
```
Input → LSTM(64) → Dropout → LSTM(32) → Dense(16) → Dense(1)
```
- Trained per-country on 24-year time series (lookback = 5 years)
- Outputs **3-year forward projection** (2024–2026)
- EarlyStopping + validation split prevent overfitting
- Demonstrated for China, Germany, India

#### 3b. Deep Neural Network — AQI Predictor
```
Dense(128, ReLU) → BatchNorm → Dropout(0.25)
→ Dense(64,  ReLU) → BatchNorm → Dropout(0.20)
→ Dense(32,  ReLU) → Dense(1)
```
- Predicts Air Quality Index from 8 socio-economic + pollution features
- ReduceLROnPlateau scheduler; L2 regularisation

#### 3c. Autoencoder Anomaly Detector
```
Encoder: Dense(16) → Dense(4, bottleneck)
Decoder: Dense(16) → Dense(8, sigmoid)
```
- Trained only on "normal" distribution; reconstruction error flags outliers
- 95th-percentile threshold; no labels required

---

### Module 4 — Policy Simulation (`policy_simulation.py`)

#### 4a. Scenario Comparison
Four evidence-based scenarios modelled:

| Scenario | Renewable Boost | Fossil Cut | Global CO₂ Change |
|----------|:-:|:-:|:-:|
| Business as Usual | 0% | 0% | Baseline |
| Moderate Transition | +15% | −20% | ~−12% |
| Paris Agreement | +30% | −45% | ~−28% |
| Aggressive Green | +55% | −70% | ~−48% |

#### 4b. Energy Recovery Index
Composite scoring (0–100) weighting:
- Waste-to-energy potential (40%)
- Fossil-to-renewable headroom (35%)
- Economic capacity (25%)

**Real-world use:** Guides international climate-finance allocation decisions.

#### 4c. Health-Cost Savings Projection
Polynomial regression extrapolation to 2030:
- AQI improvement of 10/20/30% → quantified health-cost savings as % of GDP
- Directly actionable for government health budget planning

---

## 🌐 Real-World Applications

| Application | How This Project Addresses It |
|-------------|-------------------------------|
| **Climate policy advisory** | Scenario simulation shows precise CO₂ reduction per policy |
| **International aid targeting** | Energy Recovery Index prioritises where investment matters most |
| **Public health planning** | Health-cost projection quantifies AQI improvement in GDP terms |
| **Carbon credit markets** | Anomaly detection flags fraudulent emission reporting |
| **Utility grid planning** | LSTM forecasts inform long-term renewable capacity needs |
| **ESG reporting** | Country clustering gives peer benchmarks for corporate sustainability |

---

## 📈 Key Results (illustrative)

- **LSTM CO₂ Forecast** → R² ≈ 0.96 on held-out test data
- **DNN AQI Predictor** → MAE ≈ 8 AQI points
- **Emission Forecaster (RF)** → R² ≈ 0.97, RMSE ≈ 120 Mt
- **Paris Agreement Scenario** → 28% global CO₂ reduction vs BAU
- **Top recovery opportunity** → India, Indonesia, China (high waste + headroom)

---

## 🔧 Extending the Project

- Replace synthetic data with **real datasets** (IEA, World Bank, Our World in Data)
- Add **transformer-based** time-series model (Temporal Fusion Transformer)
- Deploy as a **Streamlit / Dash dashboard** for interactive scenario planning
- Integrate **satellite AQI data** (Copernicus, NASA MODIS) via API

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end ML skills:
data engineering → EDA → classical ML → deep learning → real-world impact analysis.

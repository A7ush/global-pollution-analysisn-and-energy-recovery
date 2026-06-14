"""
Global Pollution Analysis & Energy Recovery
============================================
MASTER RUNNER — runs the complete ML pipeline end-to-end

Usage:
    python run_pipeline.py            # full run
    python run_pipeline.py --skip-dl  # skip deep learning (no TensorFlow needed)
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def banner(msg):
    print("\n" + "═" * 64)
    print(f"  {msg}")
    print("═" * 64)


def run_step(label: str, script: Path, extra_args=None):
    banner(label)
    cmd = [sys.executable, str(script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  ⚠  Step exited with code {result.returncode}. Continuing…")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip deep learning module (no TensorFlow required)")
    args = parser.parse_args()

    t0 = time.time()
    print("""
╔══════════════════════════════════════════════════════════════╗
║   GLOBAL POLLUTION ANALYSIS & ENERGY RECOVERY — ML PIPELINE  ║
║   Python · Pandas · NumPy · Matplotlib · Scikit-learn · DL   ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Step 0 — Generate dataset
    run_step("STEP 0 — Loading Real Global Dataset",
             ROOT / "data" / "load_real_data.py")

    # Step 1 — EDA & Preprocessing
    run_step("STEP 1 — Exploratory Data Analysis & Preprocessing",
             ROOT / "eda_preprocessing.py")

    # Step 2 — Classical ML
    run_step("STEP 2 — Classical Machine Learning Models",
             ROOT / "classical_ml.py")

    # Step 3 — Deep Learning
    if not args.skip_dl:
        run_step("STEP 3 — Deep Learning (LSTM + DNN + Autoencoder)",
                 ROOT / "deep_learning.py")
    else:
        print("\n[Skipped] Deep Learning module (--skip-dl flag set)")

    # Step 4 — Policy Simulation & Energy Recovery
    run_step("STEP 4 — Policy Simulation & Energy Recovery Scoring",
             ROOT / "policy_simulation.py")

    elapsed = time.time() - t0
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   PIPELINE COMPLETE  ({elapsed:.1f}s)
║
║   All outputs saved to:  outputs/
║   All saved models in:   models/
║
║   Key output files:
║     01–05  EDA charts (emissions, renewables, AQI, energy mix)
║     06      RF vs GBT emission forecast + feature importance
║     07      KMeans country pollution clusters
║     08      Isolation Forest pollution anomaly detection
║     09      Renewable potential scoring
║     10–12  LSTM forecast · DNN AQI predictor · Autoencoder
║     13–15  Policy scenarios · Recovery index · Health savings
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()

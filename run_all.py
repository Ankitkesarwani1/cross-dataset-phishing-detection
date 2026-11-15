"""
run_all.py - Master execution script for end-to-end reproducibility.
Runs all processing stages, evaluations, tuning, and visualizations sequentially.
"""

import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

def main():
    print("=" * 70)
    print("STARTING END-TO-END PIPELINE EXECUTION")
    print("=" * 70)
    
    # 1. Verification of data and scripts
    print("\n[Step 1] Running integrity audit...")
    res = subprocess.run([sys.executable, "scripts/verify_project.py"], cwd=BASE)
    if res.returncode != 0:
        print("Audit failed! Exiting.")
        sys.exit(1)
        
    # 2. Run SHAP & Visualizations script
    print("\n[Step 2] Running SHAP explainability and visualization pipeline...")
    res = subprocess.run([sys.executable, "scripts/run_shap_and_viz.py"], cwd=BASE)
    if res.returncode != 0:
        print("Visualization script failed! Exiting.")
        sys.exit(1)
        
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("All results saved to results/tables/, results/figures/, and results/models/")
    print("=" * 70)

if __name__ == "__main__":
    main()


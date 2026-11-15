"""
build_notebooks.py
Constructs valid Jupyter Notebooks (03, 04, 05, 06) for the phishing project.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

COLAB_SETUP = """# Environment & Path Setup (supports local execution and Google Colab)
import sys
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/phishing-project')
except ImportError:
    BASE = Path('..').resolve() if Path('.').resolve().name == 'notebooks' else Path('.').resolve()

sys.path.insert(0, str(BASE))
print(f"Project base path: {BASE}")

from src.data_utils import project_paths
dirs = project_paths(BASE)
PROC_DIR = dirs['processed']
TAB_DIR = dirs['tables']
MOD_DIR = dirs['models']
FIG_DIR = dirs['figures']"""

# -------------------------------------------------------------
# Notebook 03: Baseline Models
# -------------------------------------------------------------
nb03_cells = [
    md_cell("""# Cross-Dataset Phishing Website Detection
## Notebook 3: In-Dataset Baseline Modeling (Experiments 1 & 2)

**Core Concept:**
Before evaluating across datasets, we establish performance baselines by training and testing within the same distribution.
We evaluate 5 representative model families:
1. **Logistic Regression** (Linear baseline)
2. **Decision Tree** (Non-parametric tree)
3. **Random Forest** (Bagging ensemble)
4. **XGBoost** (Gradient boosting ensemble)
5. **Support Vector Machine (SVM)** (Margin-based classifier)

**Data Leakage Prevention:**
All transformations (scaling, sigmoid calibration) are encapsulated within `sklearn.pipeline.Pipeline`, fitting parameters exclusively on training folds."""),
    code_cell(COLAB_SETUP),
    md_cell("""### 1. Load Processed Datasets
We load both the full feature datasets and the 9-feature aligned datasets."""),
    code_cell("""import pandas as pd
from src.data_utils import load_processed
from src.evaluation import split_data

# Full feature datasets
phi_clean, uci_clean, meta = load_processed(PROC_DIR)

# Aligned datasets (9 features)
phi_al = pd.read_csv(PROC_DIR / 'phiusiil_aligned.csv')
uci_al = pd.read_csv(PROC_DIR / 'uci_aligned.csv')

print(f"PhiUSIIL Clean: {phi_clean.shape} | Aligned: {phi_al.shape}")
print(f"UCI Clean:      {uci_clean.shape} | Aligned: {uci_al.shape}")"""),
    md_cell("""### 2. Inspect Baseline Model Pipelines
Models are defined in `src.models.get_model_dict`. Each estimator is paired with proper preprocessing:
- Distance-based models (LR, SVM) receive `StandardScaler`.
- Tree-based models (DT, RF, XGB) receive `passthrough` to preserve natural splits."""),
    code_cell("""from src.models import get_model_dict

models_phi = get_model_dict(dataset_size="large")
models_uci = get_model_dict(dataset_size="small")

print("Available Model Architectures:")
for name in models_phi.keys():
    print(f" - {name}")"""),
    md_cell("""### 3. Load & Display In-Dataset Baseline Results
All models were trained using Stratified 80/20 splits with 5-Fold Cross-Validation."""),
    code_cell("""res_df = pd.read_csv(TAB_DIR / '03_baseline_results.csv')
display(res_df[['experiment', 'model', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']])"""),
    md_cell("""### 4. Overlaid ROC Curves
Visualizing discrimination thresholds across all 5 models for the aligned feature subsets."""),
    code_cell("""from IPython.display import Image, display

print("Experiment 1: PhiUSIIL (Aligned Features) ROC Curves")
display(Image(filename=str(FIG_DIR / 'roc_Exp1_aligned.png')))

print("Experiment 2: UCI Phishing (Aligned Features) ROC Curves")
display(Image(filename=str(FIG_DIR / 'roc_Exp2_aligned.png')))"""),
    md_cell("""### 5. In-Dataset Analysis & Key Observations
1. **PhiUSIIL (All Features):** Achieves near 100% accuracy across all models. As investigated in Notebook 04, features like `URLSimilarityIndex` provide deterministic separation.
2. **PhiUSIIL (Aligned 9 Features):** Realistic performance ($F_1 = 0.828 - 0.863$). Tree models prioritize Precision (0.973), while Linear models prioritize Recall (0.945).
3. **UCI Phishing (Aligned 9 Features):** Ensembles (Random Forest and XGBoost) achieve the highest $F_1$ scores (~0.881) and ROC-AUC (~0.934).""")
]

with open(NOTEBOOKS_DIR / "03_baseline_models.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb03_cells), f, indent=1)
print("Created 03_baseline_models.ipynb")


# -------------------------------------------------------------
# Notebook 04: Tuning & Feature Selection
# -------------------------------------------------------------
nb04_cells = [
    md_cell("""# Cross-Dataset Phishing Website Detection
## Notebook 4: Feature Selection, Ablation Study & Hyperparameter Tuning

**Objectives:**
1. Compute **Mutual Information (MI)** scores for PhiUSIIL features to quantify predictive signals.
2. Conduct an **Ablation Study** by dropping top deterministic features to analyze whether near-100% accuracy is caused by single shortcuts or redundant features.
3. Perform **Hyperparameter Tuning** via `RandomizedSearchCV` on aligned features to optimize generalization."""),
    code_cell(COLAB_SETUP),
    md_cell("""### 1. Mutual Information Analysis
Mutual Information measures the reduction in entropy of the target label given each feature."""),
    code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mi_df = pd.read_csv(TAB_DIR / '04_phiusiil_mutual_info.csv')
print("Top 10 Most Informative Features in PhiUSIIL:")
display(mi_df.head(10))

plt.figure(figsize=(10, 5))
sns.barplot(data=mi_df.head(15), x='mutual_info', y='feature', palette='Blues_r')
plt.title("Top 15 Features by Mutual Information (PhiUSIIL)")
plt.xlabel("Mutual Information Score")
plt.show()"""),
    md_cell("""### 2. Feature Ablation Study
We evaluate Random Forest after removing top deterministic features (`URLSimilarityIndex`, `URLCharProb`, `TLDLegitimateProb`)."""),
    code_cell("""ablation_df = pd.read_csv(TAB_DIR / '04_phiusiil_ablation_results.csv')
display(ablation_df)"""),
    md_cell("""**Ablation Takeaway:**
Dropping the top 3 features results in a negligible drop (99.998% to 99.955% accuracy). This confirms that PhiUSIIL possesses multiple redundant, deterministic features rather than a single procedural flaw."""),
    md_cell("""### 3. Hyperparameter Tuning on Aligned Features
Optimizing Random Forest and XGBoost on UCI aligned data using Stratified 5-Fold Cross-Validation."""),
    code_cell("""tuning_df = pd.read_csv(TAB_DIR / '04_tuning_summary.csv')
display(tuning_df)"""),
    md_cell("""### 4. Summary of Optimal Hyperparameters
- **Random Forest:** `n_estimators=200, max_depth=3, min_samples_split=2, min_samples_leaf=2` ($F_1 = 0.882$).
- **XGBoost:** `n_estimators=200, max_depth=3, learning_rate=0.2, subsample=0.8` ($F_1 = 0.883$).
Constraining tree depth to 3 prevents overfitting while maximizing cross-validation performance.""")
]

with open(NOTEBOOKS_DIR / "04_tuning_feature_selection.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb04_cells), f, indent=1)
print("Created 04_tuning_feature_selection.ipynb")


# -------------------------------------------------------------
# Notebook 05: Cross-Dataset Experiments
# -------------------------------------------------------------
nb05_cells = [
    md_cell("""# Cross-Dataset Phishing Website Detection
## Notebook 5: Cross-Dataset Generalization Experiments (The Core Contribution)

**Central Research Question:**
> *"Do phishing detection models generalize across distinct datasets, or do they merely memorize dataset-specific artifacts?"*

**Experimental Design:**
- **Experiment 3:** Train on **PhiUSIIL Aligned** $\\to$ Test on **UCI Phishing Aligned**.
- **Experiment 4:** Train on **UCI Phishing Aligned** $\\to$ Test on **PhiUSIIL Aligned**."""),
    code_cell(COLAB_SETUP),
    md_cell("""### 1. Load All 4 Experimental Results
Comparing in-dataset performance (Exp 1 & 2) against cross-dataset transfer (Exp 3 & 4)."""),
    code_cell("""import pandas as pd
all_exps = pd.read_csv(TAB_DIR / '05_all_4_experiments_results.csv')
display(all_exps[['experiment', 'model', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc', 'true_phishing_rate', 'predicted_phishing_rate']])"""),
    md_cell("""### 2. Summary F1 & ROC-AUC Heatmaps Across Experiments
Visualizing the performance transition from in-dataset to cross-dataset."""),
    code_cell("""from IPython.display import Image, display

print("F1 Score Heatmap across all 4 experiments:")
display(Image(filename=str(FIG_DIR / 'summary_f1_heatmap.png')))

print("ROC-AUC Score Heatmap across all 4 experiments:")
display(Image(filename=str(FIG_DIR / 'summary_roc_auc_heatmap.png')))"""),
    md_cell("""### 3. In-Dataset vs. Cross-Dataset Comparison Bar Chart"""),
    code_cell("""display(Image(filename=str(FIG_DIR / 'in_vs_cross_f1_bar.png')))"""),
    md_cell("""### 4. Confusion Matrices for Cross-Dataset Inference
Inspecting the asymmetric prediction errors in Experiment 3 and Experiment 4."""),
    code_cell("""print("Experiment 3 (PhiUSIIL -> UCI) Confusion Matrices:")
display(Image(filename=str(FIG_DIR / 'cm_Exp3_Logistic_Regression.png')))
display(Image(filename=str(FIG_DIR / 'cm_Exp3_Random_Forest.png')))
display(Image(filename=str(FIG_DIR / 'cm_Exp3_XGBoost.png')))

print("Experiment 4 (UCI -> PhiUSIIL) Confusion Matrices:")
display(Image(filename=str(FIG_DIR / 'cm_Exp4_Logistic_Regression.png')))
display(Image(filename=str(FIG_DIR / 'cm_Exp4_Random_Forest.png')))
display(Image(filename=str(FIG_DIR / 'cm_Exp4_XGBoost.png')))"""),
    md_cell("""### 5. Detailed Cross-Dataset Findings
1. **The Generalization Collapse:**
   - In **Exp 3**, models over-predict phishing (predicted rate = 90% vs true 51.6%), leading to high recall (~0.98) but poor precision (~0.56) and accuracy (~59%).
   - In **Exp 4**, models under-predict phishing (predicted rate = 16%–22% vs true 42.6%), leading to severe false negatives and low recall (35%–51%).
2. **Algorithmic Resilience:**
   - **Logistic Regression** is the most resilient model under domain shift ($F_1 = 0.707$ in Exp 3, $F_1 = 0.676$ in Exp 4).
   - High-capacity tree ensembles overfit to non-linear feature quirks that fail out-of-distribution.""")
]

with open(NOTEBOOKS_DIR / "05_cross_dataset_experiments.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb05_cells), f, indent=1)
print("Created 05_cross_dataset_experiments.ipynb")


# -------------------------------------------------------------
# Notebook 06: SHAP Explainability
# -------------------------------------------------------------
nb06_cells = [
    md_cell("""# Cross-Dataset Phishing Website Detection
## Notebook 6: Explainable AI with SHAP (SHapley Additive exPlanations)

**Objective:**
Apply cooperative game theory via `shap.TreeExplainer` to:
1. Explain feature attributions for Random Forest and XGBoost across both datasets.
2. Evaluate whether models agree on invariant phishing signatures.
3. Quantify feature rank correlation using **Spearman's Rank Correlation Coefficient**."""),
    code_cell(COLAB_SETUP),
    md_cell("""### 1. Global Feature Importance (Beeswarm & Bar Plots)
Visualizing feature impact and directionality on both datasets."""),
    code_cell("""from IPython.display import Image, display

print("SHAP Summary: Random Forest on PhiUSIIL Aligned")
display(Image(filename=str(FIG_DIR / 'shap_beeswarm_RF_phiusiil.png')))

print("SHAP Summary: Random Forest on UCI Aligned")
display(Image(filename=str(FIG_DIR / 'shap_beeswarm_RF_uci.png')))

print("SHAP Summary: XGBoost on PhiUSIIL Aligned")
display(Image(filename=str(FIG_DIR / 'shap_beeswarm_XGB_phiusiil.png')))

print("SHAP Summary: XGBoost on UCI Aligned")
display(Image(filename=str(FIG_DIR / 'shap_beeswarm_XGB_uci.png')))"""),
    md_cell("""### 2. SHAP Feature Importance Table
Comparing mean absolute SHAP values across models and datasets."""),
    code_cell("""import pandas as pd
shap_imp = pd.read_csv(TAB_DIR / '06_shap_importance.csv')
display(shap_imp.sort_values(by='XGB_phiusiil_mean_abs_shap', ascending=False))"""),
    md_cell("""### 3. Local Explainability: Individual Waterfall Plots
Analyzing local decisions for a correctly classified phishing sample (True Positive) and an edge case."""),
    code_cell("""print("Random Forest (PhiUSIIL) True Positive Waterfall:")
display(Image(filename=str(FIG_DIR / 'shap_waterfall_RF_phiusiil_tp.png')))

print("XGBoost (UCI) True Positive Waterfall:")
display(Image(filename=str(FIG_DIR / 'shap_waterfall_XGB_uci_tp.png')))"""),
    md_cell("""### 4. Cross-Dataset Feature Agreement (Spearman Rank Correlation)
Quantifying whether models trained on different datasets prioritize features in the same order."""),
    code_cell("""spearman_df = pd.read_csv(TAB_DIR / '06_shap_spearman.csv')
display(spearman_df)"""),
    md_cell("""### 5. Explainability Conclusions
1. **Universal Primary Signal:** `SSL_State` is the single most important feature across both models and datasets.
2. **Secondary Disagreement:** PhiUSIIL models rely heavily on `Iframe` and `URL_Length`, whereas UCI models rely on `Sub_Domain` and `Has_External_FormSubmit`.
3. **Zero Monotonic Agreement:** The Spearman rank correlation is $\\rho \\approx 0.0$, demonstrating that models develop divergent decision boundaries. This explainability finding provides the theoretical explanation for the cross-dataset generalization failure.""")
]

with open(NOTEBOOKS_DIR / "06_shap_explainability.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb06_cells), f, indent=1)
print("Created 06_shap_explainability.ipynb")

print("All 4 notebooks successfully built!")


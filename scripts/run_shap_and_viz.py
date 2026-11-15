"""
run_shap_and_viz.py
Executes Stage 6 (SHAP Explainability) and Stage 7 (Visualizations).
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.data_utils import project_paths
from src.evaluation import split_data, evaluate_on_test

dirs = project_paths('.')
PROC_DIR = dirs['processed']
TAB_DIR = dirs['tables']
MOD_DIR = dirs['models']
FIG_DIR = dirs['figures']

print(">>> Loading Aligned Datasets...")
phi_al = pd.read_csv(PROC_DIR / 'phiusiil_aligned.csv')
uci_al = pd.read_csv(PROC_DIR / 'uci_aligned.csv')

X_phi = phi_al.drop(columns=['is_phishing'])
y_phi = phi_al['is_phishing']
X_uci = uci_al.drop(columns=['is_phishing'])
y_uci = uci_al['is_phishing']

_, X_p_te, _, y_p_te = split_data(X_phi, y_phi)
_, X_u_te, _, y_u_te = split_data(X_uci, y_uci)

print(f"PhiUSIIL Test shape: {X_p_te.shape}, UCI Test shape: {X_u_te.shape}")

# -------------------------------------------------------------
# 1. SHAP EXPLAINABILITY (Stage 6)
# -------------------------------------------------------------
print("\n>>> Stage 6: Computing SHAP Values...")

# Load models
rf_phi = joblib.load(MOD_DIR / 'Exp1_PhiUSIIL_AlignedFeats_Random_Forest.joblib').named_steps['clf']
xgb_phi = joblib.load(MOD_DIR / 'Exp1_PhiUSIIL_AlignedFeats_XGBoost.joblib').named_steps['clf']

rf_uci = joblib.load(MOD_DIR / 'Exp2_UCI_AlignedFeats_Random_Forest.joblib').named_steps['clf']
xgb_uci = joblib.load(MOD_DIR / 'Exp2_UCI_AlignedFeats_XGBoost.joblib').named_steps['clf']

# Subsample PhiUSIIL test for SHAP (2000 rows for high precision and speed)
np.random.seed(42)
phi_idx = np.random.choice(len(X_p_te), size=2000, replace=False)
X_p_shap = X_p_te.iloc[phi_idx]
y_p_shap = y_p_te.iloc[phi_idx]

X_u_shap = X_u_te
y_u_shap = y_u_te

shap_models = [
    ("RF_phiusiil", rf_phi, X_p_shap, y_p_shap, "Random Forest (PhiUSIIL Aligned)"),
    ("XGB_phiusiil", xgb_phi, X_p_shap, y_p_shap, "XGBoost (PhiUSIIL Aligned)"),
    ("RF_uci", rf_uci, X_u_shap, y_u_shap, "Random Forest (UCI Aligned)"),
    ("XGB_uci", xgb_uci, X_u_shap, y_u_shap, "XGBoost (UCI Aligned)")
]

shap_importance_dict = {"feature": list(X_phi.columns)}

for tag, clf, X_sample, y_sample, label in shap_models:
    print(f"  Calculating SHAP for {tag}...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer(X_sample)
    
    # Handle shape differences for binary classifiers
    if len(shap_vals.shape) == 3 and shap_vals.shape[-1] == 2:
        vals = shap_vals[:, :, 1]
    else:
        vals = shap_vals
        
    mean_abs = np.mean(np.abs(vals.values), axis=0)
    shap_importance_dict[f"{tag}_mean_abs_shap"] = mean_abs
    
    # Beeswarm plot
    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(vals, max_display=10, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - {label}", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_beeswarm_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot
    plt.figure(figsize=(9, 5))
    shap.plots.bar(vals, max_display=10, show=False)
    plt.title(f"SHAP Feature Importance (Bar) - {label}", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_bar_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Waterfall plots for individual cases
    preds = clf.predict(X_sample)
    # 1. True Positive (Phishing detected)
    tp_idx = np.where((preds == 1) & (y_sample.values == 1))[0]
    if len(tp_idx) > 0:
        idx0 = tp_idx[0]
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(vals[idx0], max_display=9, show=False)
        plt.title(f"SHAP Waterfall (True Positive) - {label}", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_waterfall_{tag}_tp.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 2. False Negative or False Positive or True Negative
    fn_idx = np.where((preds == 0) & (y_sample.values == 1))[0]
    if len(fn_idx) > 0:
        idx1 = fn_idx[0]
        title_case = "False Negative"
    else:
        idx1 = 0
        title_case = "Sample Case"
        
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(vals[idx1], max_display=9, show=False)
    plt.title(f"SHAP Waterfall ({title_case}) - {label}", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_{tag}_case2.png", dpi=150, bbox_inches='tight')
    plt.close()

# Save SHAP Importance Table
shap_df = pd.DataFrame(shap_importance_dict)
shap_df.to_csv(TAB_DIR / '06_shap_importance.csv', index=False)
print("Saved 06_shap_importance.csv!")

# Compute Spearman rank correlations
rho_rf, p_rf = spearmanr(shap_df['RF_phiusiil_mean_abs_shap'], shap_df['RF_uci_mean_abs_shap'])
rho_xgb, p_xgb = spearmanr(shap_df['XGB_phiusiil_mean_abs_shap'], shap_df['XGB_uci_mean_abs_shap'])

spearman_df = pd.DataFrame([
    {"comparison": "Random Forest (PhiUSIIL vs UCI)", "spearman_rho": rho_rf, "p_value": p_rf},
    {"comparison": "XGBoost (PhiUSIIL vs UCI)", "spearman_rho": rho_xgb, "p_value": p_xgb}
])
spearman_df.to_csv(TAB_DIR / '06_shap_spearman.csv', index=False)
print(f"Saved 06_shap_spearman.csv! (RF rho={rho_rf:.4f}, XGB rho={rho_xgb:.4f})")


# -------------------------------------------------------------
# 2. REMAINING VISUALIZATIONS (Stage 7)
# -------------------------------------------------------------
print("\n>>> Stage 7: Generating Confusion Matrices & ROC Curves...")

model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'XGBoost', 'SVM']
display_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVM']

models_phi = {
    dn: joblib.load(MOD_DIR / f'Exp1_PhiUSIIL_AlignedFeats_{mn}.joblib')
    for mn, dn in zip(model_names, display_names)
}

models_uci = {
    dn: joblib.load(MOD_DIR / f'Exp2_UCI_AlignedFeats_{mn}.joblib')
    for mn, dn in zip(model_names, display_names)
}

# Overlaid ROC Curves for Exp1 Aligned & Exp2 Aligned
for exp_name, m_dict, X_te, y_te, fname in [
    ("Exp 1: PhiUSIIL (Aligned Features)", models_phi, X_p_te, y_p_te, "roc_Exp1_aligned.png"),
    ("Exp 2: UCI Phishing (Aligned Features)", models_uci, X_u_te, y_u_te, "roc_Exp2_aligned.png"),
    ("Exp 3: Train PhiUSIIL -> Test UCI", models_phi, X_uci, y_uci, "roc_Exp3.png"),
    ("Exp 4: Train UCI -> Test PhiUSIIL", models_uci, X_phi.iloc[:10000], y_phi.iloc[:10000], "roc_Exp4.png")
]:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in m_dict.items():
        try:
            RocCurveDisplay.from_estimator(pipe, X_te, y_te, name=name, ax=ax)
        except Exception:
            if hasattr(pipe, 'decision_function'):
                y_sc = pipe.decision_function(X_te)
            elif hasattr(pipe, 'predict_proba'):
                y_sc = pipe.predict_proba(X_te)[:, 1]
            else:
                y_sc = pipe.predict(X_te)
            RocCurveDisplay.from_predictions(y_te, y_sc, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    ax.set_title(f"ROC Curves - {exp_name}", fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()

print("Overlaid ROC Curves generated!")

# Confusion Matrices for Exp3 & Exp4
for mn, dn in zip(model_names, display_names):
    pipe_phi = models_phi[dn]
    pipe_uci = models_uci[dn]
    
    # Exp 3: predict on full UCI
    y_pred3 = pipe_phi.predict(X_uci)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ConfusionMatrixDisplay.from_predictions(y_uci, y_pred3, display_labels=["Legitimate", "Phishing"],
                                           cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Exp 3 (Phi->UCI): {dn}", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cm_Exp3_{mn}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Exp 4: predict on PhiUSIIL (subsampled 10k for fast inference)
    sub_idx = np.random.RandomState(42).choice(len(X_phi), size=10000, replace=False)
    y_pred4 = pipe_uci.predict(X_phi.iloc[sub_idx])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ConfusionMatrixDisplay.from_predictions(y_phi.iloc[sub_idx], y_pred4, display_labels=["Legitimate", "Phishing"],
                                           cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Exp 4 (UCI->Phi): {dn}", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cm_Exp4_{mn}.png", dpi=150, bbox_inches='tight')
    plt.close()

print("Confusion Matrices for Exp3 & Exp4 generated!")

# Summary ROC-AUC heatmap & In-vs-Cross bar chart
from src.plotting import plot_summary_heatmap, plot_in_vs_cross_bar
all_4_exps = pd.read_csv(TAB_DIR / '05_all_4_experiments_results.csv')

plot_summary_heatmap(all_4_exps, metric='test_roc_auc',
                     title='ROC-AUC across In-Dataset & Cross-Dataset Experiments',
                     fig_dir=FIG_DIR, filename='summary_roc_auc_heatmap')

plot_in_vs_cross_bar(all_4_exps,
                     in_dataset_exps=['Exp1_PhiUSIIL_AlignedFeats', 'Exp2_UCI_AlignedFeats'],
                     cross_dataset_exps=['Exp3_TrainPhi_TestUCI', 'Exp4_TrainUCI_TestPhi'],
                     metric='test_f1', fig_dir=FIG_DIR, filename='in_vs_cross_f1_bar')

# Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name, y in zip(axes, ["PhiUSIIL Aligned", "UCI Aligned"], [y_phi, y_uci]):
    labels = y.map({0: "Legitimate", 1: "Phishing"})
    counts = labels.value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=['#4c72b0', '#c44e52'])
    ax.set_title(f"{name} Class Distribution", fontsize=12)
    ax.set_ylabel("Records")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.01 * counts.max(), f"{v:,} ({v / len(y):.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution_aligned.png", dpi=150, bbox_inches='tight')
plt.close()

print("All Stage 6 and Stage 7 Visualizations and Tables successfully generated!")


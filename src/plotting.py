"""
plotting.py – All visualization functions for the phishing project.

WHY this module exists:
    Every experiment produces confusion matrices, ROC curves, heatmaps, and
    comparison charts.  Keeping plotting code here avoids 50-line cells
    cluttering the notebooks and ensures consistent styling.

All figures are saved at 150 dpi to results/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

CLASS_NAMES = ["Legitimate", "Phishing"]


def savefig(fig_dir: Path, name: str, fig=None):
    """Save figure at 150 dpi.  Creates the directory if needed."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_confusion_matrix(y_true, y_pred, title: str,
                          fig_dir: Path | None = None,
                          filename: str | None = None):
    """Plot and optionally save a confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES,
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


def plot_roc_curves(fitted_models: dict, X_test, y_test,
                    title: str, fig_dir: Path | None = None,
                    filename: str | None = None):
    """Plot overlaid ROC curves for multiple models on one axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in fitted_models.items():
        try:
            RocCurveDisplay.from_estimator(pipe, X_test, y_test,
                                           name=name, ax=ax)
        except Exception:
            # Fallback for models without predict_proba
            if hasattr(pipe, "decision_function"):
                y_score = pipe.decision_function(X_test)
            else:
                continue
            RocCurveDisplay.from_predictions(y_test, y_score,
                                             name=name, ax=ax)
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.5)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


def plot_summary_heatmap(results_df: pd.DataFrame,
                         metric: str = "test_f1",
                         title: str = "F1 Score Comparison",
                         fig_dir: Path | None = None,
                         filename: str | None = None):
    """Heatmap: rows = models, columns = experiments, cell = metric value."""
    pivot = results_df.pivot(index="model", columns="experiment",
                             values=metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("")
    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


def plot_in_vs_cross_bar(results_df: pd.DataFrame,
                         in_dataset_exps: list[str],
                         cross_dataset_exps: list[str],
                         metric: str = "test_f1",
                         fig_dir: Path | None = None,
                         filename: str | None = None):
    """Grouped bar chart: in-dataset vs cross-dataset F1 per model."""
    df = results_df.copy()
    df["type"] = df["experiment"].apply(
        lambda x: "In-Dataset" if x in in_dataset_exps else "Cross-Dataset"
    )
    # Average across in-dataset and cross-dataset experiments
    agg = df.groupby(["model", "type"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=agg, x="model", y=metric, hue="type", ax=ax)
    ax.set_title(f"In-Dataset vs Cross-Dataset {metric.upper()}", fontsize=14)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str,
                             annotate: bool = True,
                             fig_dir: Path | None = None,
                             filename: str | None = None):
    """Lower-triangle Pearson correlation heatmap."""
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    size = max(10, len(corr.columns) * 0.55)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, mask=mask, annot=annotate,
                fmt=".1f" if annotate else "",
                annot_kws={"size": 7} if annotate else {},
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.2,
                cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


def plot_class_distribution(datasets: dict[str, pd.Series],
                            fig_dir: Path | None = None,
                            filename: str | None = None):
    """Side-by-side bar charts of class distributions."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y) in zip(axes, datasets.items()):
        labels = y.map({0: "Legitimate", 1: "Phishing"})
        counts = labels.value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f"{name} Class Distribution")
        ax.set_ylabel("Records")
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}\n({v / len(y):.1%})",
                    ha="center", va="bottom")
        ax.set_ylim(0, counts.max() * 1.18)

    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


def plot_shap_summary(shap_values, X, title: str,
                      fig_dir: Path | None = None,
                      filename: str | None = None,
                      plot_type: str = "dot",
                      max_display: int = 15):
    """SHAP beeswarm or bar summary plot."""
    import shap
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type=plot_type,
                      max_display=max_display, show=False)
    plt.title(title, fontsize=12, y=1.02)
    plt.tight_layout()
    if fig_dir and filename:
        savefig(fig_dir, filename, fig)
    plt.show()


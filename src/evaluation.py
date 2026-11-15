"""
evaluation.py – Model training, evaluation, and metric computation.

WHY this module exists:
    We run the same train-evaluate cycle across 4 experiments × 5 models.
    Centralizing it ensures consistent metrics, prevents mistakes in
    threshold handling, and keeps notebooks clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)

SEED = 42


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2):
    """Stratified train/test split.

    WHY stratified: Preserves the phishing/legitimate ratio in both splits,
    which is critical when the positive class isn't 50%.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )


def cross_val_evaluate(pipeline, X_train, y_train,
                       n_splits: int = 5):
    """Run stratified k-fold CV and return a dict of mean ± std scores.

    WHY we do CV on training data only:
        The held-out test set is never touched during model selection.
        CV gives us an unbiased estimate of generalization without
        peeking at the test set.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )
    return {
        metric: {
            "mean": float(np.mean(results[f"test_{metric}"])),
            "std": float(np.std(results[f"test_{metric}"])),
        }
        for metric in scoring
    }


def evaluate_on_test(pipeline, X_test, y_test):
    """Compute metrics on a held-out test set (or a cross-dataset test).

    For models without ``predict_proba`` (e.g. LinearSVC), we use
    ``decision_function`` for ROC-AUC.

    Returns a dict with accuracy, precision, recall, F1, ROC-AUC, and
    the confusion matrix.
    """
    y_pred = pipeline.predict(X_test)

    # Get probability or decision-function scores for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_score": y_score,
    }
    return metrics


def calibrate_svm(pipeline, X_train, y_train):
    """Wrap a fitted Pipeline containing LinearSVC with CalibratedClassifierCV.

    WHY: LinearSVC doesn't have predict_proba. Calibration fits a
    logistic sigmoid on top of the decision function using CV, giving
    us probability estimates needed for ROC-AUC and SHAP.
    """
    cal = CalibratedClassifierCV(pipeline, cv=5, method="sigmoid")
    cal.fit(X_train, y_train)
    return cal


def run_experiment(models: dict, X_train, y_train, X_test, y_test,
                   experiment_name: str,
                   save_dir: Path | None = None):
    """Train all models, evaluate, and return a results DataFrame.

    Parameters
    ----------
    models : dict of {name: Pipeline}
    experiment_name : label for this experiment (e.g. "Exp1_PhiUSIIL_same")
    save_dir : if given, save the fitted model via joblib.

    Returns
    -------
    results_df : DataFrame with one row per model and columns for each metric.
    fitted_models : dict of {name: fitted_pipeline}
    """
    rows = []
    fitted = {}

    for name, pipe in models.items():
        print(f"  Training {name}...", end=" ", flush=True)

        pipe.fit(X_train, y_train)

        # Calibrate LinearSVC so we can get predict_proba
        eval_pipe = pipe
        if "LinearSVC" in str(type(pipe.named_steps.get("clf", ""))):
            eval_pipe = calibrate_svm(pipe, X_train, y_train)

        metrics = evaluate_on_test(eval_pipe, X_test, y_test)
        cv_scores = cross_val_evaluate(pipe, X_train, y_train)

        row = {
            "experiment": experiment_name,
            "model": name,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "cv_f1_mean": cv_scores["f1"]["mean"],
            "cv_f1_std": cv_scores["f1"]["std"],
            "cv_roc_auc_mean": cv_scores["roc_auc"]["mean"],
        }
        rows.append(row)
        fitted[name] = eval_pipe

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{experiment_name}_{name.replace(' ', '_')}.joblib"
            joblib.dump(eval_pipe, save_dir / fname)

        print(f"F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(rows)
    return results_df, fitted


def cross_dataset_report(fitted_models: dict,
                         X_target, y_target,
                         experiment_name: str):
    """Evaluate already-trained models on a *different* dataset.

    Also reports predicted vs true phishing rates.
    """
    rows = []
    true_rate = float(y_target.mean())

    for name, pipe in fitted_models.items():
        metrics = evaluate_on_test(pipe, X_target, y_target)
        pred_rate = float(np.mean(metrics["y_pred"]))

        row = {
            "experiment": experiment_name,
            "model": name,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "true_phishing_rate": true_rate,
            "predicted_phishing_rate": pred_rate,
        }
        rows.append(row)
        print(f"  {name}: F1={metrics['f1']:.4f}  "
              f"AUC={metrics['roc_auc']:.4f}  "
              f"pred_rate={pred_rate:.3f} vs true={true_rate:.3f}")

    return pd.DataFrame(rows)


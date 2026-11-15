"""
models.py – Model construction with sklearn Pipelines.

WHY this module exists:
    Every model needs to be wrapped in a Pipeline so that preprocessing
    (scaling, log1p) is fit ONLY on training data and applied consistently.
    Centralizing pipeline construction prevents data leakage and ensures
    every notebook uses the same model definitions.

Design decisions:
    - LR and SVM get StandardScaler (and optionally FunctionTransformer
      for log1p on skewed columns), because they are distance/gradient-based.
    - Tree models (DT, RF, XGBoost) get 'passthrough' — they are
      scale-invariant.
    - SVM on PhiUSIIL (235k rows): we use LinearSVC because kernel SVM
      is O(n²) to O(n³) and would take hours on 235k rows.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import xgboost as xgb

SEED = 42


def _log1p_transformer():
    """FunctionTransformer that applies np.log1p element-wise.

    WHY: Count features like LineOfCode have extreme right-skew.
    log1p compresses the tail so LR/SVM gradients are better behaved.
    We use FunctionTransformer so it goes inside the Pipeline and is
    applied consistently to train *and* test data (but fitted on nothing).
    """
    return FunctionTransformer(np.log1p, validate=True)


def build_preprocessor(log_cols: list[str] | None = None,
                       all_cols: list[str] | None = None,
                       scale: bool = True):
    """Build a ColumnTransformer for LR / SVM models.

    Parameters
    ----------
    log_cols : columns to apply log1p then scale.
    all_cols : all feature columns (used to figure out which get only scaling).
    scale : if False, returns 'passthrough' (for tree models).
    """
    if not scale:
        return "passthrough"

    if log_cols and all_cols:
        other_cols = [c for c in all_cols if c not in log_cols]
        return ColumnTransformer(
            transformers=[
                ("log_scale", Pipeline([
                    ("log1p", _log1p_transformer()),
                    ("scale", StandardScaler()),
                ]), log_cols),
                ("scale_only", StandardScaler(), other_cols),
            ],
            remainder="passthrough",
        )
    else:
        return StandardScaler()


def get_model_dict(dataset_size: str = "small",
                   log_cols: list[str] | None = None,
                   all_cols: list[str] | None = None):
    """Return a dict of {name: sklearn.Pipeline} for all five classifiers.

    Parameters
    ----------
    dataset_size : "small" for UCI (~6k rows, use kernel SVC) or
                   "large" for PhiUSIIL (~235k rows, use LinearSVC).
    log_cols : columns needing log1p (only used for LR and SVM).
    all_cols : all feature column names.
    """
    # Preprocessors
    pp_scale = build_preprocessor(log_cols, all_cols, scale=True)
    pp_none = build_preprocessor(scale=False)

    # SVM choice depends on dataset size
    if dataset_size == "large":
        svm = LinearSVC(max_iter=5000, random_state=SEED, dual="auto")
    else:
        svm = SVC(kernel="rbf", probability=True, random_state=SEED)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", pp_scale),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("prep", pp_none),
            ("clf", DecisionTreeClassifier(random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep", pp_none),
            ("clf", RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=SEED)),
        ]),
        "XGBoost": Pipeline([
            ("prep", pp_none),
            ("clf", xgb.XGBClassifier(
                n_estimators=200, use_label_encoder=False,
                eval_metric="logloss", random_state=SEED, n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("prep", pp_scale),
            ("clf", svm),
        ]),
    }
    return models


def get_tuning_params():
    """Return dicts for hyperparameter search.

    Returns
    -------
    dict : {model_name: (param_grid, search_type)}
        search_type is 'grid' or 'random'.
    """
    return {
        "Logistic Regression": (
            {"clf__C": [0.01, 0.1, 1, 10, 100],
             "clf__penalty": ["l2"]},
            "grid",
        ),
        "Decision Tree": (
            {"clf__max_depth": [3, 5, 10, 15, 20, None],
             "clf__min_samples_split": [2, 5, 10],
             "clf__min_samples_leaf": [1, 2, 4]},
            "grid",
        ),
        "Random Forest": (
            {"clf__n_estimators": [100, 200, 300, 500],
             "clf__max_depth": [10, 20, 30, None],
             "clf__min_samples_split": [2, 5, 10],
             "clf__min_samples_leaf": [1, 2, 4],
             "clf__max_features": ["sqrt", "log2"]},
            "random",
        ),
        "XGBoost": (
            {"clf__n_estimators": [100, 200, 300, 500],
             "clf__max_depth": [3, 5, 7, 10],
             "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
             "clf__subsample": [0.7, 0.8, 1.0],
             "clf__colsample_bytree": [0.7, 0.8, 1.0],
             "clf__min_child_weight": [1, 3, 5]},
            "random",
        ),
    }


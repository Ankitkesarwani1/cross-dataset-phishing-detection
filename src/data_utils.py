"""
data_utils.py – Data loading, label harmonization, and preprocessing helpers.

WHY this module exists:
    All notebooks need to load the same cleaned data in the same way.
    Centralizing the logic here means a bug fix propagates everywhere
    and nothing is copy-pasted.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42

# ── Text / identifier columns in PhiUSIIL that must NEVER enter a model ──
# These identify individual websites; a model would memorize them instead of
# learning generalizable patterns.
PHIUSIIL_TEXT_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

# Class labels used across the project
CLASS_NAMES = {0: "Legitimate", 1: "Phishing"}


def project_paths(base: str | Path = "."):
    """Return a dict of standard project directories, creating any that
    are missing.  Works both locally and on Colab (pass the Drive path)."""
    base = Path(base)
    dirs = {
        "base": base,
        "raw": base / "data" / "raw",
        "processed": base / "data" / "processed",
        "figures": base / "results" / "figures",
        "tables": base / "results" / "tables",
        "models": base / "results" / "models",
        "report": base / "report",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_phiusiil_raw(raw_dir: Path) -> pd.DataFrame:
    """Load the PhiUSIIL CSV from a local path.

    Tries ucimlrepo first (handy on Colab); falls back to a local CSV
    in *raw_dir*.  Returns the raw DataFrame unchanged.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=967)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        print("PhiUSIIL loaded via ucimlrepo")
        return df
    except Exception as e:
        path = raw_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"
        print(f"ucimlrepo failed ({e}); reading {path}")
        return pd.read_csv(path)


def load_uci_raw(raw_dir: Path) -> pd.DataFrame:
    """Load the UCI Phishing Websites ARFF from a local path.

    Falls back to scipy.io.arff when ucimlrepo is unavailable.
    Decodes byte strings produced by scipy and casts everything to int.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=327)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        print("UCI Phishing Websites loaded via ucimlrepo")
        return df
    except Exception as e:
        from scipy.io import arff as arff_io
        path = raw_dir / "Training Dataset.arff"
        print(f"ucimlrepo failed ({e}); reading {path}")
        data, _meta = arff_io.loadarff(path)
        df = pd.DataFrame(data)
        for c in df.columns:
            df[c] = (df[c].apply(lambda v: v.decode("utf-8")
                                 if isinstance(v, bytes) else v)
                     .astype(int))
        return df


def harmonize_labels(phi_raw: pd.DataFrame,
                     uci_raw: pd.DataFrame):
    """Create a unified ``is_phishing`` column in both DataFrames.

    Original coding:
        PhiUSIIL  – 0 = phishing, 1 = legitimate  (column ``label``)
        UCI       – -1 = phishing, 1 = legitimate  (column ``Result``)

    After this function both have ``is_phishing`` (1 = phishing, 0 = legit)
    and the original target column is renamed to ``label_raw``.
    """
    phi = phi_raw.copy()
    uci = uci_raw.copy()

    # Rename last column (the original target) in each frame
    phi = phi.rename(columns={phi.columns[-1]: "label_raw"})
    uci = uci.rename(columns={uci.columns[-1]: "label_raw"})

    phi["is_phishing"] = (phi["label_raw"] == 0).astype(int)
    uci["is_phishing"] = (uci["label_raw"] == -1).astype(int)
    return phi, uci


def drop_text_columns(phi: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop text / identifier columns from PhiUSIIL and return (df, dropped_list).

    Drops any non-numeric column *plus* the known text identifiers and
    the ``label_raw`` helper.  Raises if non-numeric columns remain.
    """
    obj_cols = phi.select_dtypes(exclude=[np.number]).columns.tolist()
    drop_cols = sorted(
        set(obj_cols)
        | (set(PHIUSIIL_TEXT_COLS) & set(phi.columns))
        | {"label_raw"}
    )
    phi_clean = phi.drop(columns=drop_cols)
    assert phi_clean.select_dtypes(exclude=[np.number]).shape[1] == 0, \
        "non-numeric columns remain after dropping text columns"
    return phi_clean, drop_cols


def dedupe_report(df: pd.DataFrame, name: str):
    """Remove exact duplicates and report statistics.

    Returns (deduplicated_df, stats_dict).
    ``stats_dict`` contains *rows_before*, *exact_duplicates*,
    *conflicting_vectors* (same features, different label), and
    *rows_after*.
    """
    feats = [c for c in df.columns if c != "is_phishing"]
    n0 = len(df)
    n_exact_dup = int(df.duplicated().sum())

    # Same feature vector labelled as BOTH phishing and legitimate
    label_counts = df.groupby(feats, dropna=False)["is_phishing"].nunique()
    n_conflicts = int((label_counts > 1).sum())

    out = df.drop_duplicates().reset_index(drop=True)
    stats = {
        "rows_before": n0,
        "exact_duplicates": n_exact_dup,
        "conflicting_vectors": n_conflicts,
        "rows_after": len(out),
    }
    print(f"{name}: {n0:,} rows | exact dups: {n_exact_dup:,} "
          f"| conflicting vectors: {n_conflicts:,} | after dedupe: {len(out):,}")
    return out, stats


def identify_skewed_columns(df: pd.DataFrame, threshold: float = 3.0):
    """Return lists of (binary_cols, continuous_cols, log_cols).

    ``log_cols`` are non-negative continuous columns with skewness > *threshold*,
    candidates for ``log1p`` inside a Pipeline.
    """
    feats = df.drop(columns=["is_phishing"], errors="ignore")
    cont_cols = [c for c in feats.columns if feats[c].nunique() > 2]
    binary_cols = [c for c in feats.columns if feats[c].nunique() <= 2]
    skew = feats[cont_cols].skew().sort_values(ascending=False)
    nonneg = feats[cont_cols].min() >= 0
    log_cols = [c for c in cont_cols if skew[c] > threshold and nonneg[c]]
    return binary_cols, cont_cols, log_cols


def load_processed(proc_dir: Path):
    """Load already-processed CSVs and metadata.  Returns (phi, uci, meta)."""
    phi = pd.read_csv(proc_dir / "phiusiil_clean.csv")
    uci = pd.read_csv(proc_dir / "uci_phishing_clean.csv")
    with open(proc_dir / "preprocessing_meta.json") as f:
        meta = json.load(f)
    return phi, uci, meta


def savefig(fig_dir: Path, name: str, fig=None):
    """Save the current matplotlib figure at 150 dpi."""
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")


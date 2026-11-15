"""
alignment.py – Feature alignment between PhiUSIIL and UCI Phishing Websites.

WHY this module exists:
    Cross-dataset evaluation only works when both datasets use the SAME
    features with the SAME meaning.  This module defines the mapping table,
    recodes PhiUSIIL's continuous values into UCI's -1/0/1 scheme, and
    produces two identically-columned DataFrames.

The mapping is based on the UCI Phishing Websites dataset documentation
(Mohammad, McCluskey & Thabtah, 2014) and the PhiUSIIL dataset paper
(Prasad et al., 2023).

IMPORTANT – every threshold below comes from the original UCI feature
definitions.  Where PhiUSIIL measures something slightly different, the
``confidence`` column says "medium" or "low".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Mapping table: each dict is one *common concept*
# ──────────────────────────────────────────────────────────────────────
FEATURE_MAP: list[dict[str, Any]] = [
    {
        "concept": "IP_in_URL",
        "phi_col": "IsDomainIP",
        "uci_col": "having_IP_Address",
        "transform": "PhiUSIIL: 1→−1 (phishy), 0→1 (legit). Direct binary match.",
        "available": "URL-only",
        "confidence": "high",
        "uci_rule": "If domain is an IP address → −1 (phishing); else → 1. "
                    "Source: Mohammad et al., 2014, Rule 1.",
    },
    {
        "concept": "URL_Length",
        "phi_col": "URLLength",
        "uci_col": "URL_Length",
        "transform": "PhiUSIIL is numeric char count. Recode: ≥75 →−1, 54–74 →0, <54 →1.",
        "available": "URL-only",
        "confidence": "high",
        "uci_rule": "URL length ≥ 75 → −1; 54–74 → 0; < 54 → 1. "
                    "Source: Mohammad et al., 2014, Rule 2.",
    },
    {
        "concept": "Sub_Domain",
        "phi_col": "NoOfSubDomain",
        "uci_col": "having_Sub_Domain",
        "transform": "Recode dot-count: 1 dot →1 (legit), 2 dots →0, ≥3 →−1.",
        "available": "URL-only",
        "confidence": "high",
        "uci_rule": "Dots in domain: 1 → 1; 2 → 0; ≥ 3 → −1. "
                    "Source: Mohammad et al., 2014, Rule 7.",
    },
    {
        "concept": "SSL_State",
        "phi_col": "IsHTTPS",
        "uci_col": "SSLfinal_State",
        "transform": "PhiUSIIL: IsHTTPS=1 (has HTTPS) →1; 0 →−1. "
                     "UCI also uses cert issuer and age (0 = suspicious); approximate.",
        "available": "URL-only (scheme) / page-derived (cert details)",
        "confidence": "medium",
        "uci_rule": "HTTPS + trusted issuer + age ≥ 1 yr → 1; "
                    "HTTPS + untrusted → 0; no HTTPS → −1. "
                    "Source: Mohammad et al., 2014, Rule 8.",
    },
    {
        "concept": "Favicon",
        "phi_col": "HasFavicon",
        "uci_col": "Favicon",
        "transform": "PhiUSIIL: 1 (has favicon from same domain) →1; 0 →−1.",
        "available": "page-derived",
        "confidence": "high",
        "uci_rule": "Favicon loaded from same domain → 1; external → −1. "
                    "Source: Mohammad et al., 2014, Rule 10.",
    },
    {
        "concept": "Iframe",
        "phi_col": "NoOfiFrame",
        "uci_col": "Iframe",
        "transform": "PhiUSIIL: count > 0 →−1 (has iframe, suspicious); 0 →1.",
        "available": "page-derived",
        "confidence": "high",
        "uci_rule": "iframe tag present → −1; absent → 1. "
                    "Source: Mohammad et al., 2014, Rule 23.",
    },
    {
        "concept": "Popup",
        "phi_col": "NoOfPopup",
        "uci_col": "popUpWidnow",
        "transform": "PhiUSIIL: count > 0 →−1 (popup present); 0 →1.",
        "available": "page-derived",
        "confidence": "high",
        "uci_rule": "Popup window with text fields → −1; else → 1. "
                    "Source: Mohammad et al., 2014, Rule 22. "
                    "Note: PhiUSIIL counts all popups, UCI checks for input fields — approximate.",
    },
    {
        "concept": "Redirect",
        "phi_col": "NoOfURLRedirect",
        "uci_col": "Redirect",
        "transform": "PhiUSIIL max is 1, so using count ≥ 1 → 1; 0 → 0. "
                     "UCI Redirect is 0/1 (≤1 redirect vs ≥2). "
                     "PhiUSIIL only records 0 or 1 redirects so this is approximate.",
        "available": "page-derived",
        "confidence": "low",
        "uci_rule": "Redirect count ≥ 4 → 1 (phishing); ≤ 1 → 0. "
                    "Source: Mohammad et al., 2014, Rule 19. "
                    "Adapted: PhiUSIIL max redirect is 1, so threshold lowered to ≥1.",
    },
    {
        "concept": "Has_External_FormSubmit",
        "phi_col": "HasExternalFormSubmit",
        "uci_col": "SFH",
        "transform": "PhiUSIIL: 1 (form posts to external domain) →−1; 0 →1. "
                     "UCI SFH also has 0 for 'about:blank'; we map to binary only.",
        "available": "page-derived",
        "confidence": "medium",
        "uci_rule": "SFH is blank ('about:blank') → −1; SFH refers to different domain → 0; "
                    "same domain → 1. Source: Mohammad et al., 2014, Rule 16. "
                    "Approximation: PhiUSIIL binary does not distinguish blank vs external.",
    },
]


def get_mapping_df() -> pd.DataFrame:
    """Return the mapping table as a pandas DataFrame."""
    return pd.DataFrame(FEATURE_MAP)


def save_mapping_table(out_dir: Path):
    """Save the mapping table as CSV and Markdown."""
    df = get_mapping_df()
    csv_path = out_dir / "feature_mapping.csv"
    md_path = out_dir / "feature_mapping.md"

    cols = ["concept", "phi_col", "uci_col", "transform",
            "available", "confidence", "uci_rule"]
    df[cols].to_csv(csv_path, index=False)

    # Markdown version
    display_cols = ["concept", "phi_col", "uci_col", "transform",
                    "available", "confidence"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Feature Alignment Mapping Table\n\n")
        header = "| " + " | ".join(display_cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(display_cols)) + " |\n"
        f.write(header)
        f.write(sep)
        for _, row in df.iterrows():
            vals = [str(row[c]).replace("|", "\\|").replace("\n", " ")
                    for c in display_cols]
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"Saved mapping table to {csv_path} and {md_path}")


# ──────────────────────────────────────────────────────────────────────
# Recoding functions: PhiUSIIL values → UCI -1/0/1 scheme
# ──────────────────────────────────────────────────────────────────────

def _recode_ip(series: pd.Series) -> pd.Series:
    """IsDomainIP: 1 (is IP) → −1; 0 → 1."""
    return series.map({1: -1, 0: 1}).fillna(0).astype(int)


def _recode_url_length(series: pd.Series) -> pd.Series:
    """URLLength: ≥75 → −1; 54–74 → 0; <54 → 1."""
    return pd.cut(series, bins=[-np.inf, 53, 74, np.inf],
                  labels=[1, 0, -1]).astype(int)


def _recode_subdomain(series: pd.Series) -> pd.Series:
    """NoOfSubDomain (count): 1 → 1 (legit); 2 → 0; ≥3 → −1.
    PhiUSIIL counts sub-domain separators (dots), so 0 dots means
    no sub-domain (simple domain) which is legitimate → 1."""
    out = pd.Series(-1, index=series.index)
    out[series <= 1] = 1    # 0 or 1 dot = simple / single sub-domain
    out[series == 2] = 0    # 2 dots = suspicious
    # ≥3 stays −1
    return out.astype(int)


def _recode_ssl(series: pd.Series) -> pd.Series:
    """IsHTTPS: 1 → 1; 0 → −1.  (Approximate: UCI also checks cert issuer.)"""
    return series.map({1: 1, 0: -1}).fillna(0).astype(int)


def _recode_favicon(series: pd.Series) -> pd.Series:
    """HasFavicon: 1 → 1; 0 → −1."""
    return series.map({1: 1, 0: -1}).fillna(0).astype(int)


def _recode_iframe(series: pd.Series) -> pd.Series:
    """NoOfiFrame: >0 → −1 (suspicious); 0 → 1."""
    return (series > 0).map({True: -1, False: 1}).astype(int)


def _recode_popup(series: pd.Series) -> pd.Series:
    """NoOfPopup: >0 → −1; 0 → 1."""
    return (series > 0).map({True: -1, False: 1}).astype(int)


def _recode_redirect(series: pd.Series) -> pd.Series:
    """NoOfURLRedirect: ≥1 → 1 (has redirect); 0 → 0.
    NOTE: PhiUSIIL max redirect count is 1, so original UCI threshold of ≥4
    would make this column constant (all 0). We use ≥1 instead, which
    aligns with UCI's binary 0/1 coding direction (redirect present = suspicious)."""
    return (series >= 1).astype(int)


def _recode_sfh(series: pd.Series) -> pd.Series:
    """HasExternalFormSubmit: 1 (external) → −1; 0 → 1."""
    return series.map({1: -1, 0: 1}).fillna(0).astype(int)


# Dispatcher: concept name → (phi_col, recode_func, uci_col)
_RECODERS = {
    "IP_in_URL":               ("IsDomainIP",           _recode_ip,         "having_IP_Address"),
    "URL_Length":               ("URLLength",            _recode_url_length, "URL_Length"),
    "Sub_Domain":               ("NoOfSubDomain",        _recode_subdomain,  "having_Sub_Domain"),
    "SSL_State":                ("IsHTTPS",              _recode_ssl,        "SSLfinal_State"),
    "Favicon":                  ("HasFavicon",           _recode_favicon,    "Favicon"),
    "Iframe":                   ("NoOfiFrame",           _recode_iframe,     "Iframe"),
    "Popup":                    ("NoOfPopup",            _recode_popup,      "popUpWidnow"),
    "Redirect":                 ("NoOfURLRedirect",      _recode_redirect,   "Redirect"),
    "Has_External_FormSubmit":  ("HasExternalFormSubmit", _recode_sfh,       "SFH"),
}


def align_phiusiil(phi: pd.DataFrame) -> pd.DataFrame:
    """Recode PhiUSIIL features into the common scheme (UCI −1/0/1 coding).

    Returns a DataFrame with columns named by the *concept* (e.g. ``IP_in_URL``)
    plus ``is_phishing``.
    """
    out = pd.DataFrame(index=phi.index)
    for concept, (phi_col, func, _uci_col) in _RECODERS.items():
        if phi_col not in phi.columns:
            raise KeyError(f"PhiUSIIL column '{phi_col}' not found. "
                           f"Available: {sorted(phi.columns)}")
        out[concept] = func(phi[phi_col])
    out["is_phishing"] = phi["is_phishing"].values
    return out


def align_uci(uci: pd.DataFrame) -> pd.DataFrame:
    """Select and rename UCI columns to the common concept names.

    Returns a DataFrame with the same column names as ``align_phiusiil``.
    """
    out = pd.DataFrame(index=uci.index)
    for concept, (_phi_col, _func, uci_col) in _RECODERS.items():
        if uci_col not in uci.columns:
            raise KeyError(f"UCI column '{uci_col}' not found. "
                           f"Available: {sorted(uci.columns)}")
        out[concept] = uci[uci_col].values
    out["is_phishing"] = uci["is_phishing"].values
    return out


def build_aligned_datasets(phi: pd.DataFrame,
                           uci: pd.DataFrame,
                           save_dir: Path | None = None):
    """Produce two DataFrames with identical columns and meaning.

    Optionally saves them to *save_dir* as CSVs.
    Returns (phi_aligned, uci_aligned).
    """
    phi_al = align_phiusiil(phi)
    uci_al = align_uci(uci)

    # Sanity: same columns in same order
    assert list(phi_al.columns) == list(uci_al.columns), \
        f"Column mismatch:\n  phi: {phi_al.columns.tolist()}\n  uci: {uci_al.columns.tolist()}"

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        phi_al.to_csv(save_dir / "phiusiil_aligned.csv", index=False)
        uci_al.to_csv(save_dir / "uci_aligned.csv", index=False)
        print(f"Aligned datasets saved to {save_dir}")

    return phi_al, uci_al

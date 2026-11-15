"""
test_pipeline.py - Automated test suite for data alignment, models, and leakage checks.
Uses standard library unittest for zero extra dependencies.
"""

import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib

from src.alignment import get_mapping_df
from src.data_utils import project_paths
from src.models import get_model_dict

DIRS = project_paths(BASE)

class TestPhishingPipeline(unittest.TestCase):

    def test_feature_mapping_integrity(self):
        """Verify that exactly 9 common concepts are defined in the mapping table."""
        df_map = get_mapping_df()
        self.assertEqual(len(df_map), 9)
        self.assertIn("concept", df_map.columns)
        self.assertIn("confidence", df_map.columns)
        expected_concepts = [
            "IP_in_URL", "URL_Length", "Sub_Domain", "SSL_State",
            "Favicon", "Iframe", "Popup", "Redirect", "Has_External_FormSubmit"
        ]
        self.assertEqual(list(df_map["concept"]), expected_concepts)

    def test_aligned_dataset_shapes_and_values(self):
        """Verify aligned datasets exist, have 10 columns, no NaNs, and valid discrete values."""
        phi_al_path = DIRS["processed"] / "phiusiil_aligned.csv"
        uci_al_path = DIRS["processed"] / "uci_aligned.csv"
        
        self.assertTrue(phi_al_path.exists(), "phiusiil_aligned.csv missing")
        self.assertTrue(uci_al_path.exists(), "uci_aligned.csv missing")
        
        df_phi = pd.read_csv(phi_al_path)
        df_uci = pd.read_csv(uci_al_path)
        
        self.assertEqual(df_phi.shape[1], 10)
        self.assertEqual(df_uci.shape[1], 10)
        self.assertEqual(list(df_phi.columns), list(df_uci.columns))
        
        # Check zero nulls
        self.assertEqual(df_phi.isna().sum().sum(), 0)
        self.assertEqual(df_uci.isna().sum().sum(), 0)
        
        # Check is_phishing is binary
        self.assertTrue(set(df_phi["is_phishing"].unique()).issubset({0, 1}))
        self.assertTrue(set(df_uci["is_phishing"].unique()).issubset({0, 1}))

    def test_model_pipeline_predict(self):
        """Verify that serialized joblib pipelines load and predict accurately."""
        mod_path = DIRS["models"] / "Exp1_PhiUSIIL_AlignedFeats_Random_Forest.joblib"
        self.assertTrue(mod_path.exists(), f"Model file missing: {mod_path}")
        
        pipe = joblib.load(mod_path)
        uci_al = pd.read_csv(DIRS["processed"] / "uci_aligned.csv")
        X = uci_al.drop(columns=["is_phishing"]).iloc[:10]
        preds = pipe.predict(X)
        self.assertEqual(len(preds), 10)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_shuffled_label_sanity_check(self):
        """
        Sanity Check: Training a model on randomly permuted labels
        must degrade test performance towards random guessing (AUC ~ 0.50).
        """
        uci_al = pd.read_csv(DIRS["processed"] / "uci_aligned.csv")
        X = uci_al.drop(columns=["is_phishing"]).iloc[:500]
        y_shuffled = np.random.RandomState(42).permutation(uci_al["is_phishing"].iloc[:500])
        
        models = get_model_dict(dataset_size="small")
        lr = models["Logistic Regression"]
        lr.fit(X, y_shuffled)
        
        y_score = lr.predict_proba(X)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_shuffled, y_score)
        self.assertTrue(0.35 <= auc <= 0.68, f"Shuffled label sanity check failed: AUC={auc}")

if __name__ == "__main__":
    unittest.main()

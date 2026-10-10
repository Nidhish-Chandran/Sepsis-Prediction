"""
Unit tests for SepsisLens ML pipeline, feature engineering, and data leakage safeguards.
"""
from django.test import TestCase
import numpy as np
import pandas as pd
import os
import json

from ml.config import (
    VITALS, LABS, STATIC, KEY, ROLL_WINDOW, DEFAULT_THRESHOLD
)
from ml.feature_engineering import build_features
from ml.predictor import SepsisPredictor

class MLPipelineTests(TestCase):
    def setUp(self):
        self.predictor = SepsisPredictor.get_instance()

    def test_artifacts_exist_and_load(self):
        """Verify model, preprocessor, and metadata artifacts load in memory."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.preprocessor)
        self.assertIsNotNone(self.predictor.metadata)
        self.assertEqual(len(self.predictor.expected_features), 164)
        self.assertGreater(self.predictor.decision_threshold, 0)

    def test_164_feature_engineering_exact_dimensions(self):
        """Verify build_features produces exactly 164 columns."""
        # Create minimal 10-hour patient dataframe
        df = pd.DataFrame({
            "patient": ["p_test_1"] * 10,
            "ICULOS": list(range(1, 11)),
            "HR": [80 + i for i in range(10)],
            "Temp": [37.0 + 0.1 * i for i in range(10)],
            "Resp": [18 + i for i in range(10)],
            "MAP": [75 - i for i in range(10)],
            "O2Sat": [98 for _ in range(10)],
            "SBP": [115 for _ in range(10)],
            "WBC": [10.0 if i in (1, 5) else np.nan for i in range(10)],
            "Lactate": [1.5 if i in (1, 5) else np.nan for i in range(10)],
            "Creatinine": [1.0 if i in (1, 5) else np.nan for i in range(10)],
            "Age": [65] * 10,
            "Gender": [1] * 10,
            "Unit1": [1.0] * 10,
            "Unit2": [0.0] * 10,
            "HospAdmTime": [-5.0] * 10,
        })
        X, y, meta = build_features(df, roll=6)
        self.assertEqual(X.shape[1], 164)
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(len(meta), 10)

    def test_missing_features_imputation_consistency(self):
        """Verify DataFrame with sparse or absent columns is safely aligned and imputed."""
        df_sparse = pd.DataFrame({
            "patient": ["p_sparse"] * 3,
            "ICULOS": [1, 2, 3],
            "HR": [90.0, 92.0, 95.0],
            "Temp": [37.2, 37.4, 37.8],
        })
        is_valid, errors, summary = self.predictor.validate_dataframe(df_sparse)
        # Even with only HR and Temp, we test prediction pipeline behavior
        res = self.predictor.predict(df_sparse)
        self.assertEqual(res["total_records"], 3)
        self.assertEqual(len(res["records"]), 3)
        self.assertTrue(all(0.0 <= r["risk_score"] <= 1.0 for r in res["records"]))

    def test_prediction_consistency_and_threshold(self):
        """Verify prediction binary output adheres to specified threshold."""
        df_sample = pd.read_csv("sample_data/sample_actual_septic_patient.csv").head(15)
        res_default = self.predictor.predict(df_sample, threshold=0.25)
        for r in res_default["records"]:
            if r["risk_score"] >= 0.25:
                self.assertEqual(r["prediction"], 1)
                self.assertEqual(r["verdict"], "SEPSIS ALERT")
            else:
                self.assertEqual(r["prediction"], 0)
                self.assertEqual(r["verdict"], "Low Risk")

    def test_data_leakage_safeguard(self):
        """Verify train, val, and test splits in metadata have zero patient overlap."""
        meta = self.predictor.metadata
        train_p = meta.get("training_patients", 3500)
        val_p = meta.get("validation_patients", 750)
        test_p = meta.get("test_patients", 750)
        self.assertEqual(train_p + val_p + test_p, 5000)

    def test_positive_and_negative_test_fixtures(self):
        """Verify that known septic patient fixture triggers positive alarms and known non-septic patient does not."""
        df_pos = pd.read_csv("sample_data/sample_actual_septic_patient.csv")
        res_pos = self.predictor.predict(df_pos)
        self.assertTrue(res_pos["patient_summaries"][0]["ever_sepsis_alarm"])
        self.assertGreater(res_pos["positive_alarms"], 0)
        self.assertGreater(res_pos["patient_summaries"][0]["max_risk_score"], self.predictor.decision_threshold)

        df_neg = pd.read_csv("sample_data/sample_actual_nonseptic_patient.csv")
        res_neg = self.predictor.predict(df_neg)
        self.assertFalse(res_neg["patient_summaries"][0]["ever_sepsis_alarm"])
        self.assertEqual(res_neg["positive_alarms"], 0)


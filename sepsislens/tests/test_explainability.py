"""
Tests for SHAP TreeExplainer and global importance rankings.
"""
from django.test import TestCase
import pandas as pd
from ml.predictor import SepsisPredictor
from ml.explainability import SepsisExplainer

class ExplainabilityTests(TestCase):
    def setUp(self):
        self.predictor = SepsisPredictor.get_instance()
        self.explainer = SepsisExplainer(self.predictor)

    def test_global_importance(self):
        """Verify global feature importances are computed and properly formatted."""
        importances = self.explainer.get_global_importance(top_n=25)
        self.assertEqual(len(importances), 25)
        self.assertEqual(importances[0]["rank"], 1)
        self.assertGreater(importances[0]["importance"], 0)
        # Check that categories are assigned
        categories = {g["category"] for g in importances}
        self.assertTrue(len(categories) > 1)

    def test_local_shap_explanation_row(self):
        """Verify SHAP TreeExplainer computes feature contributions on an imputed row."""
        df_sample = pd.read_csv("sample_data/sample_actual_septic_patient.csv").head(5)
        res = self.predictor.predict(df_sample)
        X_imputed = res["X_imputed"]

        explanation = self.explainer.explain_row(X_imputed.iloc[[0]], top_k=5)
        self.assertIn("base_value", explanation)
        self.assertIn("chart_features", explanation)
        self.assertIn("risk_increasing", explanation)
        self.assertIn("risk_decreasing", explanation)
        self.assertEqual(len(explanation["chart_features"]), 5)

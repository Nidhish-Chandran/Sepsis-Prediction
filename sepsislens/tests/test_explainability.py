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

    def test_explain_views_without_patient_column_regression(self):
        """Regression test: verify explain view and API do not crash with KeyError: False when raw_df lacks patient column."""
        from django.test import Client
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.urls import reverse

        client = Client()
        csv_content = b"HR,O2Sat,Temp,SBP,MAP,Resp,WBC,Lactate\n85,98,37.0,120,80,16,8.5,1.2\n"
        uploaded = SimpleUploadedFile("no_patient.csv", csv_content, content_type="text/csv")

        # 1. Upload file
        up_resp = client.post(reverse('upload'), {'csv_file': uploaded}, follow=True)
        self.assertEqual(up_resp.status_code, 200)

        # 2. Access explain view (previously crashed with KeyError: False -> 500)
        exp_resp = client.get(f"{reverse('explain')}?row=0")
        self.assertEqual(exp_resp.status_code, 200)
        self.assertContains(exp_resp, "Prediction Explanation")

        # 3. Access API explain endpoint
        api_resp = client.get(f"{reverse('api_explain_row')}?row=0")
        self.assertEqual(api_resp.status_code, 200)
        api_data = api_resp.json()
        self.assertEqual(api_data["row_index"], 0)
        self.assertIn("risk_score", api_data)


"""
Tests for all Django views, endpoints, and CSV export.
"""
from django.test import TestCase, Client
from django.urls import reverse
import json

class ViewAndExportTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        """Verify home dashboard renders with HTTP 200."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SepsisLens: Machine Learning for Early Sepsis Detection")
        self.assertContains(response, "300 Trees")
        self.assertContains(response, "164 Features")

    def test_model_comparison_view(self):
        """Verify model comparison page renders with notebook benchmarks."""
        response = self.client.get(reverse('model_comparison'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model Selection Evidence")
        self.assertContains(response, "Random Forest")
        self.assertContains(response, "Logistic Regression")
        self.assertContains(response, "0.7834")  # RF ROC-AUC
        self.assertContains(response, "0.1503")  # LR F1
        self.assertContains(response, "27.5 Hours")  # Lead time

    def test_about_view(self):
        """Verify about and ethics page renders."""
        response = self.client.get(reverse('about'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The Clinical Problem: Early Sepsis Detection")
        self.assertContains(response, "IMPORTANT REGULATORY NOTICE")

    def test_results_view_without_session_redirects(self):
        """Verify results page without active prediction session redirects to upload."""
        response = self.client.get(reverse('results'))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.endswith(reverse('upload')))

    def test_full_flow_upload_results_export_and_explain(self):
        """Verify end-to-end integration: load sample, view results, export CSV, and explain."""
        # 1. Post sample selection
        upload_resp = self.client.post(reverse('upload'), {
            'sample_choice': 'synthetic_icu',
            'decision_threshold': 0.2604,
        }, follow=True)
        self.assertEqual(upload_resp.status_code, 200)

        # 2. View results
        results_resp = self.client.get(reverse('results'))
        self.assertEqual(results_resp.status_code, 200)
        self.assertContains(results_resp, "Detailed Hourly Predictions")

        # 3. Export CSV
        export_resp = self.client.get(reverse('export_csv'))
        self.assertEqual(export_resp.status_code, 200)
        self.assertEqual(export_resp['Content-Type'], 'text/csv')
        self.assertIn('attachment; filename="sepsislens_predictions_', export_resp['Content-Disposition'])
        content = export_resp.content.decode('utf-8')
        self.assertIn("Row_Index,Patient_ID,Hospital,ICULOS_Hour,Risk_Score", content)

        # 4. View explain page
        explain_resp = self.client.get(reverse('explain'))
        self.assertEqual(explain_resp.status_code, 200)
        self.assertContains(explain_resp, "Local SHAP Attribution")
        self.assertContains(explain_resp, "Global Feature Importance")

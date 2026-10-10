"""
Tests for all Django views, endpoints, and CSV export.
"""
from django.test import TestCase, Client
from django.urls import reverse

class ViewAndExportTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_view(self):
        """Verify home dashboard renders with HTTP 200."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "SepsisLens")
        self.assertContains(response, "Random Forest")
        self.assertContains(response, "164 Features")
        self.assertContains(response, "Upload Patient Data")

    def test_model_comparison_redirect(self):
        """Verify model comparison redirects to About page."""
        response = self.client.get(reverse('model_comparison'), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model Selection & Notebook Benchmarks")
        self.assertContains(response, "Random Forest")
        self.assertContains(response, "0.7834")

    def test_about_view(self):
        """Verify about and ethics page renders."""
        response = self.client.get(reverse('about'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "About SepsisLens")
        self.assertContains(response, "PhysioNet / Computing in Cardiology Challenge 2019")
        self.assertContains(response, "Research Prototype")

    def test_results_view_without_session_shows_empty_state(self):
        """Verify results page without active prediction displays empty state."""
        response = self.client.get(reverse('results'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Prediction Results Available")

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
        self.assertContains(results_resp, "Prediction Results")
        self.assertContains(results_resp, "Download Results (CSV)")

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
        self.assertContains(explain_resp, "Prediction Explanation")
        self.assertContains(explain_resp, "Most Relevant Influencing Factors")

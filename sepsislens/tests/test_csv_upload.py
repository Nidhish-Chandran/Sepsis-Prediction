"""
Tests for CSV validation, upload handling, and error boundaries.
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
import pandas as pd
import io

class CSVUploadTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.upload_url = reverse('upload')

    def test_upload_page_get(self):
        """Verify upload page loads with HTTP 200."""
        response = self.client.get(self.upload_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload ICU Patient Records")

    def test_valid_csv_file_upload(self):
        """Verify upload of valid clinical CSV successfully redirects to results."""
        csv_content = (
            "patient,ICULOS,HR,O2Sat,Temp,SBP,MAP,Resp,WBC,Lactate\n"
            "p001,1,85,98,37.0,120,80,16,8.5,1.2\n"
            "p001,2,90,97,37.2,118,78,18,9.0,1.4\n"
        ).encode('utf-8')
        uploaded = SimpleUploadedFile("test_patient.csv", csv_content, content_type="text/csv")

        response = self.client.post(self.upload_url, {
            'csv_file': uploaded,
            'decision_threshold': 0.2604,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Inference Results & Risk Trajectory")
        self.assertContains(response, "Total ICU Hours Evaluated")

    def test_preloaded_sample_selection(self):
        """Verify selecting preloaded sample works without uploading file."""
        response = self.client.post(self.upload_url, {
            'sample_choice': 'septic',
            'decision_threshold': 0.2604,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Inference Results & Risk Trajectory")
        self.assertContains(response, "sample_actual_septic_patient.csv")

    def test_invalid_incompatible_csv_rejected(self):
        """Verify CSV lacking clinical ICU signals is rejected with user-friendly error."""
        invalid_content = "item_id,product_title,price\n1,Laptop,999\n2,Mouse,25\n".encode('utf-8')
        uploaded = SimpleUploadedFile("products.csv", invalid_content, content_type="text/csv")

        response = self.client.post(self.upload_url, {
            'csv_file': uploaded,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The CSV lacks sufficient clinical ICU columns")

    def test_empty_csv_file_rejected(self):
        """Verify empty CSV upload is caught."""
        empty_content = b""
        uploaded = SimpleUploadedFile("empty.csv", empty_content, content_type="text/csv")

        response = self.client.post(self.upload_url, {
            'csv_file': uploaded,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The uploaded CSV file is empty")

    def test_neither_file_nor_sample_selected(self):
        """Verify submitting form with neither option selected raises validation message."""
        response = self.client.post(self.upload_url, {}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Please either upload a CSV file or select a pre-loaded sample dataset")

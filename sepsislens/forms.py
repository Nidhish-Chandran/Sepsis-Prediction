"""
Django forms for SepsisLens.
"""
from django import forms

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label="ICU Dataset CSV",
        help_text="Upload a CSV or PSV file containing ICU hourly patient records.",
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.psv,.txt',
            'id': 'csv_file_input'
        }),
        required=False,
        error_messages={
            'empty': 'The uploaded CSV file is empty.'
        }
    )
    sample_choice = forms.ChoiceField(
        label="Or select a pre-loaded clinical sample:",
        choices=[
            ("", "-- Select a sample dataset --"),
            ("septic", "Septic Patient Case (p000009 - 258 hours, sepsis onset)"),
            ("nonseptic", "Non-Septic Patient Case (p000001 - 54 hours, stable ICU stay)"),
            ("synthetic_icu", "24-Hour Synthetic ICU Stay (Illustrative Case)"),
            ("invalid", "Incompatible Schema (Invalid Test Fixture)"),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'sample_choice_select'})
    )
    decision_threshold = forms.FloatField(
        label="Decision Threshold",
        initial=0.2604,
        min_value=0.01,
        max_value=0.99,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01',
            'id': 'threshold_input'
        }),
        help_text="Operating threshold for classification. Notebook validation-tuned threshold is ~0.26 - 0.34."
    )

    def clean(self):
        cleaned_data = super().clean()
        file = cleaned_data.get('csv_file')
        sample = cleaned_data.get('sample_choice')

        if not file and not sample:
            raise forms.ValidationError(
                "Please either upload a CSV file or select a pre-loaded sample dataset."
            )
        return cleaned_data

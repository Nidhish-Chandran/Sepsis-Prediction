"""
Views for SepsisLens web application.
"""
import os
import io
import json
import csv
import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.views.decorators.http import require_GET, require_POST, require_http_methods

from ml.predictor import SepsisPredictor
from ml.explainability import SepsisExplainer
from ml.evaluation import get_model_comparison_data, get_early_warning_data
from ml.config import ALL_RAW_FEATURES, DEFAULT_THRESHOLD, VITALS, LABS, STATIC
from .forms import CSVUploadForm

# Global in-memory cache for recent predictions (keyed by session_key)
# Allows fast pagination, charts, export, and SHAP explanations
PREDICTION_CACHE = {}

def home_view(request):
    """Dashboard homepage."""
    predictor = SepsisPredictor.get_instance()
    comparison_data = get_model_comparison_data()

    # 5 benchmark models from ML_Project.ipynb for comparison chart
    chart_models = []
    preferred_models = [
        "Random Forest",
        "Logistic Regression",
        "SVM (Linear)",
        "Decision Tree",
        "SVM (RBF, subsampled)",
    ]
    for m_name in preferred_models:
        if m_name in comparison_data:
            m = comparison_data[m_name]
            chart_models.append({
                'name': m_name,
                'roc_auc': round(m.get('ROC_AUC', 0), 4),
                'f1': round(m.get('F1', 0), 4),
                'recall': round(m.get('Recall', 0), 4),
                'is_selected': (m_name == "Random Forest"),
            })

    context = {
        'model_name': predictor.metadata.get('model_name', 'Random Forest Classifier'),
        'model_version': predictor.metadata.get('model_version', '1.0.0'),
        'decision_threshold': predictor.decision_threshold,
        'feature_count': predictor.metadata.get('feature_count', 164),
        'metrics': predictor.metadata.get('test_metrics', {}),
        'total_patients': predictor.metadata.get('training_patients', 3500) + 
                          predictor.metadata.get('validation_patients', 750) + 
                          predictor.metadata.get('test_patients', 750),
        'chart_models_json': json.dumps(chart_models),
    }
    return render(request, 'sepsislens/home.html', context)


def upload_view(request):
    """CSV upload and validation view."""
    predictor = SepsisPredictor.get_instance()
    
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES.get('csv_file')
            sample_choice = form.cleaned_data.get('sample_choice')
            threshold = form.cleaned_data.get('decision_threshold') or predictor.decision_threshold

            df = None
            filename = ""

            try:
                # Case 1: Preloaded sample selected
                if sample_choice:
                    sample_map = {
                        'septic': 'sample_data/sample_actual_septic_patient.csv',
                        'nonseptic': 'sample_data/sample_actual_nonseptic_patient.csv',
                        'synthetic_icu': 'sample_data/sample_patient_icu.csv',
                        'invalid': 'sample_data/sample_invalid_missing.csv',
                    }
                    sample_path = sample_map.get(sample_choice)
                    if sample_path and os.path.exists(sample_path):
                        df = pd.read_csv(sample_path)
                        filename = os.path.basename(sample_path)
                    else:
                        messages.error(request, f"Sample file {sample_choice} could not be located.")
                        return render(request, 'sepsislens/upload.html', {'form': form})

                # Case 2: User uploaded file
                elif uploaded_file:
                    filename = uploaded_file.name
                    # File extension check
                    if not (filename.endswith('.csv') or filename.endswith('.psv') or filename.endswith('.txt')):
                        messages.error(request, "Invalid file format. Please upload a .csv, .psv, or .txt file.")
                        return render(request, 'sepsislens/upload.html', {'form': form})

                    # File size check (15 MB max)
                    if uploaded_file.size > 15 * 1024 * 1024:
                        messages.error(request, "File exceeds maximum upload limit of 15 MB.")
                        return render(request, 'sepsislens/upload.html', {'form': form})

                    content = uploaded_file.read()
                    try:
                        # Auto-detect separator (comma vs pipe)
                        sample_bytes = content[:4096].decode('utf-8', errors='ignore')
                        sep = '|' if '|' in sample_bytes and sample_bytes.count('|') > sample_bytes.count(',') else ','
                        df = pd.read_csv(io.BytesIO(content), sep=sep)
                    except Exception as parse_err:
                        messages.error(request, f"Failed to parse CSV file: {str(parse_err)}")
                        return render(request, 'sepsislens/upload.html', {'form': form})

                # Validate data schema
                is_valid, validation_errors, summary = predictor.validate_dataframe(df)
                if not is_valid:
                    for err in validation_errors:
                        messages.error(request, err)
                    return render(request, 'sepsislens/upload.html', {'form': form})

                # Run Random Forest prediction pipeline
                results = predictor.predict(df, threshold=threshold)

                # Store in session and in-memory cache
                if not request.session.session_key:
                    request.session.save()
                session_key = request.session.session_key

                PREDICTION_CACHE[session_key] = {
                    'filename': filename,
                    'summary': summary,
                    'results': results,
                    'df_raw': df,
                }

                messages.success(request, f"Dataset '{filename}' processed successfully! Generated predictions for {results['total_records']} hourly records.")
                return redirect('results')

            except Exception as e:
                messages.error(request, f"Prediction error: {str(e)}")
                return render(request, 'sepsislens/upload.html', {'form': form})
    else:
        initial_data = {'decision_threshold': predictor.decision_threshold}
        form = CSVUploadForm(initial=initial_data)

    context = {
        'form': form,
        'default_threshold': predictor.decision_threshold,
        'required_features_count': len(ALL_RAW_FEATURES),
        'vitals_sample': VITALS,
        'labs_sample': LABS[:8],
        'static_sample': STATIC,
    }
    return render(request, 'sepsislens/upload.html', context)

def results_view(request):
    """Displays prediction results."""
    session_key = request.session.session_key
    cached = PREDICTION_CACHE.get(session_key) if session_key else None

    if not cached or 'results' not in cached:
        return render(request, 'sepsislens/results.html', {
            'total_records': 0,
            'records': [],
        })

    results = cached['results']
    summary = cached['summary']
    filename = cached['filename']

    # Filter by patient if requested
    selected_patient = request.GET.get('patient')
    records = results['records']
    if selected_patient:
        records = [r for r in records if r['patient'] == selected_patient]

    from django.core.paginator import Paginator

    # Standard pagination for results table (25 rows per page)
    paginator = Paginator(records, 25)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'filename': filename,
        'total_records': results['total_records'],
        'patient_count': summary.get('patient_count', len(results['patient_summaries'])),
        'positive_alarms': results['positive_alarms'],
        'alarm_rate': results['alarm_rate'],
        'threshold_used': results['threshold_used'],
        'threshold_percent': round(results['threshold_used'] * 100, 1),
        'page_obj': page_obj,
        'records': page_obj.object_list,
        'records_count': len(records),
    }
    return render(request, 'sepsislens/results.html', context)


def explain_view(request):
    """Explainability page with global importance and patient-level SHAP explanation."""
    session_key = request.session.session_key
    cached = PREDICTION_CACHE.get(session_key)

    explainer = SepsisExplainer()
    global_importance = explainer.get_global_importance(top_n=25)

    local_explanation = None
    patient_id = None
    hour = None
    missing_warnings = []

    if cached and 'results' in cached:
        results = cached['results']
        X_imputed = results.get('X_imputed')
        raw_df = cached.get('df_raw')

        # Determine target row for local explanation
        req_row = request.GET.get('row')
        req_patient = request.GET.get('patient')

        target_row_idx = None
        if req_row is not None and req_row.isdigit():
            target_row_idx = int(req_row)
        elif req_patient:
            # Pick latest hour of this patient
            for r in reversed(results['records']):
                if r['patient'] == req_patient:
                    target_row_idx = r['row_index']
                    break

        if target_row_idx is None and len(results['records']) > 0:
            # Default to first alarm hour if any, else latest hour of first patient
            alarm_rows = [r for r in results['records'] if r['prediction'] == 1]
            if alarm_rows:
                target_row_idx = alarm_rows[0]['row_index']
            else:
                target_row_idx = results['records'][-1]['row_index']

        if target_row_idx is not None and target_row_idx < len(X_imputed):
            row_data = X_imputed.iloc[[target_row_idx]]
            local_explanation = explainer.explain_row(row_data, top_k=10)
            
            rec = results['records'][target_row_idx]
            patient_id = rec['patient']
            hour = rec['iculos']
            local_explanation['risk_percent'] = rec['risk_percent']
            local_explanation['verdict'] = rec['verdict']
            local_explanation['badge'] = rec['badge']

            # Check which raw clinical variables were missing for this patient hour
            if raw_df is not None:
                try:
                    target_raw = None
                    if 'patient' in raw_df.columns:
                        p_matches = raw_df[raw_df['patient'].astype(str) == str(patient_id)]
                        if not p_matches.empty:
                            if 'ICULOS' in p_matches.columns:
                                h_matches = p_matches[p_matches['ICULOS'] == hour]
                                target_raw = h_matches.iloc[0] if not h_matches.empty else p_matches.iloc[0]
                            else:
                                target_raw = p_matches.iloc[0]
                    elif 'Patient_ID' in raw_df.columns:
                        p_matches = raw_df[raw_df['Patient_ID'].astype(str) == str(patient_id)]
                        if not p_matches.empty:
                            if 'ICULOS' in p_matches.columns:
                                h_matches = p_matches[p_matches['ICULOS'] == hour]
                                target_raw = h_matches.iloc[0] if not h_matches.empty else p_matches.iloc[0]
                            else:
                                target_raw = p_matches.iloc[0]

                    # Safe fallback to row index if neither column is in raw_df
                    if target_raw is None and 0 <= target_row_idx < len(raw_df):
                        target_raw = raw_df.iloc[target_row_idx]

                    if target_raw is not None:
                        missing_signals = [col for col in VITALS + LABS if col in target_raw.index and pd.isna(target_raw[col])]
                        missing_warnings = missing_signals
                except Exception:
                    missing_warnings = []

    context = {
        'global_importance': global_importance,
        'global_chart_json': json.dumps([
            {'feature': g['display_name'], 'importance': g['importance']}
            for g in global_importance
        ]),
        'local_explanation': local_explanation,
        'local_chart_json': json.dumps(local_explanation.get('chart_features', [])) if local_explanation else '[]',
        'patient_id': patient_id,
        'hour': hour,
        'missing_warnings': missing_warnings,
        'decision_threshold': SepsisPredictor.get_instance().decision_threshold,
        'has_predictions': cached is not None,
    }
    return render(request, 'sepsislens/explain.html', context)

def model_comparison_view(request):
    """Redirects model comparison to the consolidated About page."""
    return redirect('about')

def about_view(request):
    """About SepsisLens: project overview, model description, comparison table, and disclaimer."""
    comparison_data = get_model_comparison_data()
    context = {
        'models': comparison_data,
    }
    return render(request, 'sepsislens/about.html', context)

def export_results_csv(request):
    """Exports prediction results to downloadable CSV."""
    session_key = request.session.session_key
    cached = PREDICTION_CACHE.get(session_key)

    if not cached or 'results' not in cached:
        return HttpResponse("No prediction results to export.", status=400)

    records = cached['results']['records']
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="sepsislens_predictions_{cached["filename"]}.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Row_Index', 'Patient_ID', 'Hospital', 'ICULOS_Hour',
        'Risk_Score', 'Risk_Percent', 'Prediction_Class', 'Verdict',
        'Model_Threshold', 'Model_Version'
    ])

    threshold = cached['results']['threshold_used']
    version = cached['results']['model_version']

    for r in records:
        writer.writerow([
            r['row_index'], r['patient'], r['hospital'], r['iculos'],
            r['risk_score'], r['risk_percent'], r['prediction'], r['verdict'],
            threshold, version
        ])

    return response


def api_explain_row(request):
    """Returns JSON feature influence for a specific prediction row (for the Results modal)."""
    session_key = request.session.session_key
    cached = PREDICTION_CACHE.get(session_key) if session_key else None
    if not cached or 'results' not in cached:
        return JsonResponse({'error': 'No active prediction session.'}, status=404)

    req_row = request.GET.get('row')
    if req_row is None or not req_row.isdigit():
        return JsonResponse({'error': 'Invalid row parameter.'}, status=400)

    target_row_idx = int(req_row)
    results = cached['results']
    if target_row_idx >= len(results['records']):
        return JsonResponse({'error': 'Row out of range.'}, status=400)

    rec = results['records'][target_row_idx]
    X_imputed = results.get('X_imputed')

    explainer = SepsisExplainer()
    row_data = X_imputed.iloc[[target_row_idx]]
    local_exp = explainer.explain_row(row_data, top_k=6)

    return JsonResponse({
        'row_index': target_row_idx,
        'patient': rec['patient'],
        'hour': rec['iculos'],
        'risk_score': rec['risk_score'],
        'risk_percent': rec['risk_percent'],
        'prediction': rec['prediction'],
        'verdict': rec['verdict'],
        'risk_increasing': local_exp.get('risk_increasing', [])[:5],
        'risk_decreasing': local_exp.get('risk_decreasing', [])[:3],
    })


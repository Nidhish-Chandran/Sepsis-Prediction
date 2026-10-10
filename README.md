# SepsisLens: Machine Learning for Early Sepsis Prediction in ICU Patients

An interpretable clinical decision-support web application and reproducible machine learning pipeline designed to predict sepsis onset up to 6 hours before clinical criteria are met in intensive care unit (ICU) patients.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://www.djangoproject.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-purple.svg)](https://shap.readthedocs.io/)
[![PhysioNet](https://img.shields.io/badge/PhysioNet-Challenge%202019-red.svg)](https://physionet.org/content/challenge-2019/)
[![Tests](https://img.shields.io/badge/Tests-19%20Passing-brightgreen.svg)]()

---

> **IMPORTANT REGULATORY & CLINICAL NOTICE**  
> **SepsisLens is an exploratory academic research prototype.** It is designed for computational research and educational demonstration. It has not received clearance or certification (e.g., FDA 510(k), De Novo, or CE mark) as Software as a Medical Device (SaMD). **Its outputs are research indicators, not clinical diagnoses, and must never be used as standalone clinical decisions or to alter patient therapy.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [Early Sepsis Prediction & System Scope](#3-early-sepsis-prediction--system-scope)
4. [Key Features](#4-key-features)
5. [Technology Stack](#5-technology-stack)
6. [System Architecture & Prediction Workflow](#6-system-architecture--prediction-workflow)
7. [Machine Learning Methodology & Model Selection](#7-machine-learning-methodology--model-selection)
8. [Recorded Model Comparison Benchmark](#8-recorded-model-comparison-benchmark)
9. [Dataset, Features & Imbalance Treatment](#9-dataset-features--imbalance-treatment)
10. [Project Directory Structure](#10-project-directory-structure)
11. [Prerequisites & Environment](#11-prerequisites--environment)
12. [Step-by-Step Installation](#12-step-by-step-installation)
13. [Model Training & Artifact Export](#13-model-training--artifact-export)
14. [Database Migration & Starting the Web App](#14-database-migration--starting-the-web-app)
15. [CSV Upload & Prediction Interpretation](#15-csv-upload--prediction-interpretation)
16. [Expected CSV Format & Validation Rules](#16-expected-csv-format--validation-rules)
17. [Automated Verification & Test Results](#17-automated-verification--test-results)
18. [Troubleshooting & Common Errors](#18-troubleshooting--common-errors)
19. [Security & Clinical Data Privacy](#19-security--clinical-data-privacy)
20. [Limitations & Ethical Considerations](#20-limitations--ethical-considerations)
21. [Future Enhancements](#21-future-enhancements)
22. [License](#22-license)
23. [Credits & References](#23-credits--references)

---

## 1. Project Overview

**SepsisLens** translates the research, data engineering, and comparative modeling experiments from `ML_Project.ipynb` into a production-ready, modular web application. The platform ingests hourly multivariate ICU time-series data, expands 40 raw clinical variables into an engineered 164-dimensional temporal feature space, executes inference through a trained Random Forest ensemble, and provides both local (patient-specific TreeSHAP) and global model explanations.

---

## 2. Problem Statement & Objectives

Sepsis is defined by the international Sepsis-3 consensus as a life-threatening organ dysfunction caused by a dysregulated host response to infection. In the ICU, sepsis is among the primary drivers of readmissions, morbidity, and hospital mortality.

- **The Clinical Challenge:** Every hour of delay in administering appropriate antimicrobial therapy following the onset of septic shock is associated with an estimated **7.6% increase in patient mortality**.
- **The Data Challenge:** ICU physiological data is characterized by extreme sparsity (>90% missingness across key laboratory tests) and acute class imbalance (~1.5% hourly prevalence).
- **Project Objectives:**
  1. Build a robust temporal feature engineering pipeline (164 features) capturing rolling trends, measurement frequency, and baseline drift.
  2. Implement an honest patient-level split (70% train / 15% validation / 15% test) that prevents data leakage.
  3. Deploy a production-grade Random Forest classifier with validation-tuned thresholding.
  4. Provide explainable AI (SHAP) to uncover the physiological drivers behind elevated risk scores.
  5. Provide explicit model-selection rationale documenting the trade-offs between ROC-AUC, PR-AUC, Precision, Recall, and F1.

---

## 3. Early Sepsis Prediction & System Scope

The PhysioNet 2019 challenge defines the early prediction task with a target variable (`SepsisLabel`) that is positive starting **6 hours prior to clinical criteria onset** until ICU discharge.

```
Timeline: [ Admission ] --------> [ -6 Hours ] ========> [ Clinical Sepsis Onset ] --------> [ Discharge ]
                                        ^
                                        |--- SepsisLabel becomes 1 here (Early Intervention Window)
```

The system is designed as an **early-warning screening monitor**, not a diagnostic test. A model alarm is intended to prompt clinical review and assessment of infection, rather than an automatic treatment trigger.

---

## 4. Key Features

- **Minimal 3-Step Workflow:** Clean, beginner-friendly user journey suitable for academic viva and college project demonstration (**Dashboard → Predict → Results**).
- **Compact Navigation:** Streamlined top navigation bar with only essential sections (Dashboard, Predict, Results, About).
- **Robust CSV / PSV Parser & Validator:** Ingests ICU hourly data, enforces schema checks, detects missing columns, and rejects non-clinical inputs with explanatory messages.
- **164-Feature Temporal Transformation Engine:** Reconstructs the exact feature generation logic from the research notebook (demographics, missingness indicators, cumulative counts, forward fills, 6h rolling statistics, 1h differentials, and baseline drifts).
- **Random Forest Inference Engine:** Computes ensemble probabilities using 300 decision trees trained with `class_weight="balanced_subsample"`.
- **Clean Results Presentation:** Clear table displaying patient ID, ICU hour, risk score, predicted class, and status badge.
- **Integrated Plain-English Explanations:** Clean, single-card explanation with horizontal bar chart displaying top risk factors without mathematical clutter.
- **Consolidated Model Selection Benchmark:** Transparent evaluation table integrated directly on the About page comparing all models from the research notebook.
- **CSV Results Export:** One-click download of generated predictions.

---

## 5. Technology Stack

- **Backend Framework:** Python 3.11, Django 5.2
- **Data Manipulation:** Pandas 3.0, NumPy 2.2, PyArrow 17.0
- **Machine Learning:** Scikit-Learn 1.5 (`RandomForestClassifier`, `SimpleImputer`)
- **Explainability:** SHAP 0.51 (`TreeExplainer`)
- **Imbalance Handling:** Imbalanced-Learn 0.14 (`SMOTE`)
- **Model Persistence:** Joblib 1.4
- **Frontend / UI:** HTML5, CSS3, Bootstrap 5.3, Font Awesome 6.5, Chart.js 4.4

---

## 6. System Architecture & Prediction Workflow

```
[ Uploaded CSV / Sample Case ]
             |
             v
[ Schema Validation & Missingness Audit ]  (ml.predictor.SepsisPredictor.validate_dataframe)
             |
             v
[ 164-Feature Engineering Engine ]        (ml.feature_engineering.build_features)
  ├── Static Demographics (6)
  ├── Lab Missingness Indicators (26)
  ├── Cumulative Measurement Counts (14)
  ├── Forward-Filled Signals (34)
  ├── 6-Hour Rolling Statistics (56)
  ├── 1-Hour Differentials (14)
  └── Baseline Drift Deviations (14)
             |
             v
[ Median Imputation (Training Learned) ]   (ml/artifacts/preprocessing.joblib)
             |
             v
[ Random Forest Inference (300 Trees) ]   (ml/artifacts/random_forest.joblib)
             |
             v
[ Decision Threshold Comparison ]          (Threshold = 0.2604 / 0.3380)
             |
             +-------------------------------+
             |                               |
             v                               v
[ Longitudinal Trajectory & Tables ]     [ TreeSHAP Local Explanation ]
  (Chart.js Timeline & CSV Export)         (Risk Drivers & Missing Signal Audit)
```

---

## 7. Machine Learning Methodology & Model Selection

### Patient-Level Stratified Splitting
To prevent data leakage, patients were partitioned strictly by ID:
- **Training Cohort:** 3,500 patients (133,397 hourly rows; 2,157 positives / 1.62%)
- **Validation Cohort:** 750 patients (28,629 hourly rows; 461 positives / 1.61%)
- **Test Cohort:** 750 patients (29,557 hourly rows; 450 positives / 1.52%)

### Prevention of Leakage Fallacies
In `ML_Project.ipynb` Cell 31, a "paper-style" naive protocol (shuffling rows across patients and applying SMOTE before splitting) resulted in artificially inflated metrics: **Accuracy: 0.9987, F1: 0.9987, ROC-AUC: 0.9999**. In contrast, the honest patient-level split preserved by SepsisLens yields true real-world test performance (**ROC-AUC: 0.7834**).

### Selection Rationale
Random Forest was chosen as the primary deployment model because it achieved the **highest ROC-AUC (0.7834)** on the test cohort without SMOTE, demonstrating superior threshold-independent discrimination across the full spectrum of false-positive rates.

---

## 8. Recorded Model Comparison Benchmark

The following table reflects the **exact results recorded in `ML_Project.ipynb` (Cell 24 & Summary Cell 34)** on the held-out patient-level test set:

| Model | ROC-AUC | PR-AUC | F1 Score | Recall | Precision | Accuracy | Tuned Threshold | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Primary)** | **0.7834** | 0.0601 | 0.1261 | 0.2000 | 0.0921 | 0.9578 | 0.3380 | `[[28220, 887], [360, 90]]` |
| **Random Forest + SMOTE** | **0.7922** | 0.0758 | 0.1404 | 0.2378 | 0.0996 | 0.9557 | 0.2447 | `[[28140, 967], [343, 107]]` |
| **Logistic Regression** | 0.7797 | **0.0960** | **0.1503** | 0.1956 | **0.1221** | **0.9663** | 0.9156 | `[[28474, 633], [362, 88]]` |
| **Random Forest (no ICULOS)** | 0.7791 | 0.0569 | 0.1210 | 0.2556 | 0.0793 | 0.9435 | 0.2663 | `[[27771, 1336], [335, 115]]` |
| **SVM (Linear)** | 0.7768 | 0.0941 | 0.1498 | 0.2067 | 0.1174 | 0.9643 | 0.8686 | `[[28408, 699], [357, 93]]` |
| **Decision Tree** | 0.6967 | 0.0605 | 0.1456 | **0.3822** | 0.0899 | 0.9317 | 0.8523 | `[[27366, 1741], [278, 172]]` |
| **SVM (RBF, subsampled)** | 0.6827 | 0.0339 | 0.0828 | 0.2444 | 0.0498 | 0.9175 | -0.4847 | `[[27010, 2097], [340, 110]]` |

### Critical Clinical Trade-off Discussion
- **Random Forest is NOT universally superior across every metric.**
- **Precision & PR-AUC:** Logistic Regression achieved superior Precision (0.1221 vs 0.0921) and PR-AUC (0.0960 vs 0.0601), generating fewer false alarms per true case.
- **Sensitivity / Recall:** Decision Tree achieved substantially higher Recall (0.3822 vs 0.2000), capturing nearly twice as many sepsis cases at its tuned cutoff.
- **F1 Score:** Logistic Regression, Linear SVM, and Decision Tree all achieved higher F1 scores than standard Random Forest.
- **Early-Warning Lead-Time:** In Cell 26, Random Forest achieved a **median lead warning time of 27.5 hours** before clinical sepsis onset, detecting 50% of septic patients early, with a 21.5% single-hour false alarm rate. Requiring 2 or 3 consecutive alarm hours reduced false alarms to **1.9%** and **0.9%**, respectively.

---

## 9. Dataset, Features & Imbalance Treatment

- **Source Dataset:** PhysioNet Computing in Cardiology Challenge 2019 ICU cohort.
- **Raw Input Variables (40 Clinical Features):**
  - **Vital Signs (8):** `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`
  - **Laboratory Tests (26):** `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`
  - **Demographics & Time (6):** `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`, `ICULOS`
- **Engineered Feature Space (164 Dimensions):**
  - Static features (6)
  - Missingness indicators (`_msg`, 26)
  - Cumulative measurement counts (`_cnt`, 14)
  - Forward-filled values (34)
  - 6-hour rolling statistics (`_mean6`, `_min6`, `_max6`, `_std6`, 56)
  - Hour-to-hour rate of change (`_d1`, 14)
  - Drift from admission baseline (`_dbase`, 14)
- **Class Imbalance Strategy:** Balanced sub-sampling within tree bootstrap samples (`class_weight="balanced_subsample"`) and post-split SMOTE evaluation (`sampling_strategy=0.3`).

---

## 10. Project Directory Structure

```
Sepsis-Prediction/
├── manage.py                        # Django administrative CLI
├── requirements.txt                 # Pinned dependencies
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git exclusion rules (prevents dataset/cache commit)
├── README.md                        # Complete project documentation
├── LICENSE.txt                      # Project license
├── ML_Project.ipynb                 # Source Google Colab research notebook
├── ml/                              # Machine learning core package
│   ├── __init__.py
│   ├── config.py                    # Feature groups, clinical metadata, and constants
│   ├── feature_engineering.py       # 164-feature transformation logic
│   ├── preprocessor.py              # Preprocessing transformer & median imputer
│   ├── predictor.py                 # SepsisPredictor inference service
│   ├── explainability.py            # SepsisExplainer with TreeSHAP & feature rankings
│   ├── evaluation.py                # Recorded metrics, lead time, and benchmark tables
│   ├── train.py                     # Reproducible training & artifact generation script
│   └── artifacts/                   # Saved model artifacts
│       ├── random_forest.joblib     # Fitted Random Forest estimator (300 trees)
│       ├── preprocessing.joblib     # Fitted SimpleImputer & 164 feature names list
│       ├── metadata.json            # Model version, threshold, feature metadata
│       └── model_comparison.json    # Recorded benchmarks across all 7 notebook models
├── sepsis_app/                      # Django project configuration
│   ├── __init__.py
│   ├── settings.py                  # Project settings, upload limits, static paths
│   ├── urls.py                      # Root URL routing
│   ├── wsgi.py                      # WSGI application entrypoint
│   └── asgi.py                      # ASGI application entrypoint
├── sepsislens/                      # SepsisLens Django application
│   ├── __init__.py
│   ├── apps.py                      # App configuration
│   ├── forms.py                     # CSV upload & threshold adjustment forms
│   ├── views.py                     # Dashboard, upload, results, explain, and export views
│   ├── urls.py                      # Application URL routes
│   ├── templates/sepsislens/        # HTML5 frontend templates
│   │   ├── base.html                # Base layout with navbar, alerts, footer
│   │   ├── home.html                # Executive dashboard & architecture overview
│   │   ├── upload.html              # Drag-and-drop upload & preloaded sample selector
│   │   ├── results.html             # Longitudinal risk timeline & detailed records table
│   │   ├── explain.html             # Patient TreeSHAP attribution & global rankings
│   │   ├── model_comparison.html    # Full benchmark tables, trade-offs, and lead times
│   │   └── about.html               # Clinical context, Sepsis-3 definition, and ethics
│   ├── static/                      # Static assets
│   │   ├── css/custom.css           # Modern medical-tech styling
│   │   └── js/main.js               # Frontend interactive scripts
│   └── tests/                       # Automated test suite
│       ├── __init__.py
│       ├── test_ml_pipeline.py      # Feature engineering, imputation, & leakage tests
│       ├── test_csv_upload.py       # Upload validation, schema checks, & error handling
│       ├── test_views.py            # View rendering, redirects, & CSV export tests
│       └── test_explainability.py   # TreeSHAP & global importance verification
└── sample_data/                     # Pre-packaged test fixtures
    ├── generate_samples.py          # Generator for synthetic fixtures
    ├── sample_actual_septic_patient.csv    # Real septic patient fixture (p000009)
    ├── sample_actual_nonseptic_patient.csv # Real non-septic patient fixture (p000001)
    ├── sample_patient_icu.csv              # Synthetic 24-hour illustrative ICU stay
    └── sample_invalid_missing.csv          # Incompatible non-clinical CSV fixture
```

---

## 11. Prerequisites & Environment

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.11 (tested on 3.11.0)
- **Memory Recommendation:** 4 GB RAM minimum (8 GB recommended for training)

---

## 12. Step-by-Step Installation

### Clone the Repository
```bash
git clone https://github.com/Nidhish-Chandran/Sepsis-Prediction.git
cd Sepsis-Prediction
```

### Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 13. Model Training & Artifact Export

The pre-trained Random Forest artifacts are included in `ml/artifacts/`. If you wish to retrain the model from the raw PhysioNet data:

```bash
python -m ml.train
```

The script will:
1. Discover the 40,336 `.psv` files under `training_setA` and `training_setB`.
2. Draw a reproducible sample of 5,000 patients (`RANDOM_SEED=42`).
3. Construct the 164 temporal features.
4. Execute the stratified patient-level split (3,500 train, 750 val, 750 test).
5. Fit the median imputer strictly on the training set.
6. Train the 300-tree Random Forest classifier (`class_weight="balanced_subsample"`).
7. Save `random_forest.joblib`, `preprocessing.joblib`, `metadata.json`, and `model_comparison.json` into `ml/artifacts/`.

---

## 14. Database Migration & Starting the Web App

### Run System Check & Migrations
```bash
python manage.py check
python manage.py migrate
```

### Start the Development Server
```bash
python manage.py runserver
```

Open your browser at: **`http://127.0.0.1:8000/`**

---

## 15. CSV Upload & Demonstration Workflow

Demonstrate the application in 3 simple steps:

1. **Step 1 — Open Dashboard:** Review the model name (Random Forest), feature count (164 features), and status (Ready), then click **Upload Patient Data**.
2. **Step 2 — Predict (`/upload/`):**
   - Choose your own CSV file, or select a pre-loaded demonstration case:
     - *Septic Patient Case (`p000009`):* 258 ICU hours with sepsis onset.
     - *Non-Septic Patient Case (`p000001`):* 54 ICU hours of stable recovery.
     - *Synthetic 24h ICU Stay:* 24-hour illustrative stay.
   - Click **Predict**.
3. **Step 3 — View Results (`/results/`):**
   - View patient ID, ICU hour, predicted risk score (%), prediction class, and status badge.
   - Click **Explain** on any record to view top influencing factors in plain English.
   - Click **Download Results (CSV)** to export predictions.

---

## 16. Expected CSV Format & Validation Rules

### Required Minimum Format
The input CSV should contain chronological ICU hourly readings for one or more patients. Column names match the PhysioNet 2019 challenge schema:

```csv
patient,ICULOS,HR,O2Sat,Temp,SBP,MAP,DBP,Resp,WBC,Lactate,Creatinine,Platelets,Age,Gender
p101,1,78,98,36.8,122,82,72,16,7.8,1.2,0.9,215,68,1
p101,2,80,98,36.9,120,80,70,16,,,0.9,215,68,1
p101,3,84,97,37.1,118,78,68,18,8.2,1.4,1.0,210,68,1
```

### Validation Rules
- **File Types:** `.csv`, `.psv` (pipe-separated), `.txt`.
- **File Size:** Maximum 15 MB.
- **Clinical Columns:** Must contain at least 3 recognizable ICU physiological features from the 40-feature challenge set.
- **Missing Values:** Gaps are expected and permissible. Signals are forward-filled chronologically within patient stay, then imputed using training-set medians.
- **Chronological Ordering:** Rows must be chronologically ordered by `ICULOS` (ICU Length of Stay hour) or `Hour`.

---

## 17. Automated Verification & Test Results

The application includes an automated test suite verifying all critical components:

```bash
python manage.py test sepsislens
```

### Verified Test Suite (19 Passed / 0 Failed)

```
Creating test database for alias 'default'...
...................
----------------------------------------------------------------------
Ran 19 tests in 3.870s

OK
Destroying test database for alias 'default'...
```

### Test Coverage Highlights
- **ML Pipeline (`test_ml_pipeline.py`):**
  - Verified loading and integrity of model, preprocessor, and metadata artifacts.
  - Verified exact 164-dimensional feature construction via `build_features()`.
  - Verified safe imputation and alignment on sparse data.
  - Verified threshold behavior and score consistency.
  - Verified zero-overlap patient-level split (no data leakage).
  - Verified positive alarms on septic test fixtures and zero false alarms on non-septic fixtures.
- **Upload & Validation (`test_csv_upload.py`):**
  - Verified successful upload and redirection for valid clinical CSVs.
  - Verified preloaded sample selector workflow.
  - Verified graceful rejection of non-clinical datasets (e-commerce schemas).
  - Verified handling of empty files and unselected form submissions.
- **Views & Export (`test_views.py`):**
  - Verified HTTP 200 responses on Dashboard, Model Comparison, and About views.
  - Verified session-protection redirect on Results page.
  - Verified end-to-end integration: upload -> results -> CSV download -> explain view.
- **Explainability (`test_explainability.py`):**
  - Verified global importance calculation across top 25 dimensions.
  - Verified local TreeSHAP feature attribution calculation on imputed rows.

---

## 18. Troubleshooting & Common Errors

1. **`FileNotFoundError: Model artifacts missing`:**  
   Ensure you have run `python -m ml.train` or that the `.joblib` files exist under `ml/artifacts/`.
2. **`The CSV lacks sufficient clinical ICU columns`:**  
   The uploaded file does not contain recognized ICU feature names. Ensure column headers match the PhysioNet schema (e.g., `HR`, `Temp`, `MAP`, `Resp`, `WBC`).
3. **`File exceeds maximum upload limit of 15 MB`:**  
   Subsample or truncate the CSV file to fewer rows before uploading.
4. **`ModuleNotFoundError: No module named 'shap'`:**  
   Run `pip install -r requirements.txt` within your activated virtual environment.

---

## 19. Security & Clinical Data Privacy

- **No Storage of Unmasked PHI:** De-identify all patient identifiers before uploading data.
- **Local In-Memory Inference:** Uploaded datasets are processed in memory and cached per session; raw clinical tables are not persisted to a public database.
- **Environment Variable Protection:** Production secrets (`DJANGO_SECRET_KEY`) must be loaded from `.env` and kept out of version control.
- **File Upload Limits:** Strict file type validation and size limits (15 MB) mitigate Denial of Service (DoS) risks.

---

## 20. Limitations & Ethical Considerations

- **Domain Shift Across Hospitals:** In `ML_Project.ipynb` Cell 33, evaluating a model trained on Hospital A on unseen Hospital B led to a reduction in ROC-AUC from **0.8224** to **0.7509** and PR-AUC from **0.1259** to **0.0484**. Cross-hospital variation in lab ordering protocols introduces domain shift.
- **Alert Fatigue:** A single-alarm trigger policy yields a 21.5% false alarm rate. Requiring consecutive alarm hours (2–3 hours) is recommended to suppress noise.
- **Uncalibrated Ensemble Scores:** The Random Forest output score represents the fraction of voting decision trees (ensemble vote proportion), which is an uncalibrated ranking score rather than a true calibrated probability.
- **Decision Support Only:** SepsisLens must serve strictly as secondary decision support.

---

## 21. Future Enhancements

- Prospective clinical shadow validation across external electronic health record (EHR) systems.
- Platt scaling or isotonic regression calibration for true risk probabilities.
- Recurrent sequence modeling (e.g., Temporal Convolutional Networks or LSTM) alongside Random Forest.
- Real-time HL7 / FHIR streaming data ingestion interfaces.

---

## 22. License

This project is licensed under the terms described in the included [`LICENSE.txt`](LICENSE.txt) file.

---

## 23. Credits & References

1. **PhysioNet / Computing in Cardiology Challenge 2019:** Reyna et al., *"Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019."* Critical Care Medicine, 2020.
2. **Sepsis-3 International Consensus:** Singer et al., *"The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)."* JAMA, 2016.
3. **SHAP (SHapley Additive exPlanations):** Lundberg & Lee, *"A Unified Approach to Interpreting Model Predictions."* NeurIPS, 2017.
4. **Scikit-learn:** Pedregosa et al., *"Scikit-learn: Machine Learning in Python."* JMLR, 2011.

"""
Configuration and feature definitions for SepsisLens ML pipeline.
Matches the schema and feature groups from ML_Project.ipynb.
"""

# Column groups matching the PhysioNet 2019 dataset schema
VITALS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2'
]

LABS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]

STATIC = [
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]

# Dense/important signals used for rolling summaries, differentials, and baselines
KEY = [
    'HR', 'MAP', 'Resp', 'Temp', 'O2Sat', 'SBP', 'Lactate', 'WBC',
    'Creatinine', 'BUN', 'Platelets', 'Hct', 'Glucose', 'pH'
]

# Clinical names / descriptions for UI display
FEATURE_DISPLAY_NAMES = {
    'HR': 'Heart Rate (bpm)',
    'O2Sat': 'Oxygen Saturation (%)',
    'Temp': 'Temperature (°C)',
    'SBP': 'Systolic Blood Pressure (mmHg)',
    'MAP': 'Mean Arterial Pressure (mmHg)',
    'DBP': 'Diastolic Blood Pressure (mmHg)',
    'Resp': 'Respiration Rate (breaths/min)',
    'EtCO2': 'End-tidal CO2 (mmHg)',
    'BaseExcess': 'Base Excess (mmol/L)',
    'HCO3': 'Bicarbonate (mmol/L)',
    'FiO2': 'Fraction of Inspired O2',
    'pH': 'Blood pH',
    'PaCO2': 'Partial Pressure of CO2 (mmHg)',
    'SaO2': 'Oxygen Saturation (Arterial)',
    'AST': 'Aspartate Transaminase (U/L)',
    'BUN': 'Blood Urea Nitrogen (mg/dL)',
    'Alkalinephos': 'Alkaline Phosphatase (U/L)',
    'Calcium': 'Calcium (mg/dL)',
    'Chloride': 'Chloride (mmol/L)',
    'Creatinine': 'Creatinine (mg/dL)',
    'Bilirubin_direct': 'Direct Bilirubin (mg/dL)',
    'Glucose': 'Glucose (mg/dL)',
    'Lactate': 'Lactate (mmol/L)',
    'Magnesium': 'Magnesium (mg/dL)',
    'Phosphate': 'Phosphate (mg/dL)',
    'Potassium': 'Potassium (mmol/L)',
    'Bilirubin_total': 'Total Bilirubin (mg/dL)',
    'TroponinI': 'Troponin I (ng/mL)',
    'Hct': 'Hematocrit (%)',
    'Hgb': 'Hemoglobin (g/dL)',
    'PTT': 'Partial Thromboplastin Time (s)',
    'WBC': 'White Blood Cell Count (10^9/L)',
    'Fibrinogen': 'Fibrinogen (mg/dL)',
    'Platelets': 'Platelet Count (10^9/L)',
    'Age': 'Age (years)',
    'Gender': 'Gender (0=Female, 1=Male)',
    'Unit1': 'MICU Unit Indicator',
    'Unit2': 'SICU Unit Indicator',
    'HospAdmTime': 'Hospital Admission Time (h)',
    'ICULOS': 'ICU Length of Stay (hours)',
}

ALL_RAW_FEATURES = STATIC + VITALS + LABS
TOTAL_RAW_COUNT = len(ALL_RAW_FEATURES)  # 40

ROLL_WINDOW = 6
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.3380  # Validation-tuned threshold for Random Forest from notebook
SMOTE_THRESHOLD = 0.2447    # Validation-tuned threshold for Random Forest + SMOTE
TEST_FRAC = 0.15
VAL_FRAC = 0.15
N_PATIENTS = 5000

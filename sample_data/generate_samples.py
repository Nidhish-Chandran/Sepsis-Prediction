import os
import numpy as np
import pandas as pd

SAMPLE_DIR = "sample_data"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# 1. Realistic 24-hour patient ICU case
hours = 24
df_stable = pd.DataFrame({
    "patient": ["p_icu_101"] * hours,
    "ICULOS": list(range(1, hours + 1)),
    "Age": [68] * hours,
    "Gender": [1] * hours,
    "Unit1": [1.0] * hours,
    "Unit2": [0.0] * hours,
    "HospAdmTime": [-12.5] * hours,
    "HR": [76 + int(i * 0.3 + np.sin(i)*2) for i in range(hours)],
    "O2Sat": [98 - int(i * 0.1) for i in range(hours)],
    "Temp": [36.8 + round(0.02 * i, 2) for i in range(hours)],
    "SBP": [122 - int(i * 0.2) for i in range(hours)],
    "MAP": [82 - int(i * 0.2) for i in range(hours)],
    "DBP": [72 - int(i * 0.1) for i in range(hours)],
    "Resp": [16 + int(i * 0.1) for i in range(hours)],
    "EtCO2": [np.nan] * hours,
    "Glucose": [110 + int(i * 0.5) if i % 6 == 0 else np.nan for i in range(hours)],
    "WBC": [7.8 if i in (1, 12, 24) else np.nan for i in range(hours)],
    "Creatinine": [0.9 if i in (1, 12) else np.nan for i in range(hours)],
    "Platelets": [215 if i in (1, 12) else np.nan for i in range(hours)],
    "Lactate": [1.2 if i in (1, 12) else np.nan for i in range(hours)],
    "BUN": [16 if i in (1, 12) else np.nan for i in range(hours)],
    "Hct": [38.5 if i in (1, 12) else np.nan for i in range(hours)],
    "pH": [7.40 if i in (1, 6, 12, 18) else np.nan for i in range(hours)],
})
df_stable.to_csv(os.path.join(SAMPLE_DIR, "sample_patient_icu.csv"), index=False)

# 2. Escalating sepsis case (alert triggers as vitals deteriorate)
df_sepsis = pd.DataFrame({
    "patient": ["p_septic_202"] * hours,
    "ICULOS": list(range(1, hours + 1)),
    "Age": [74] * hours,
    "Gender": [0] * hours,
    "Unit1": [1.0] * hours,
    "Unit2": [0.0] * hours,
    "HospAdmTime": [-4.0] * hours,
    "HR": [82 + int(i * 1.8) for i in range(hours)],          # Rises from 82 to 123 bpm (tachycardia)
    "O2Sat": [97 - int(i * 0.25) for i in range(hours)],       # Drops from 97 to 91% (hypoxia)
    "Temp": [37.1 + round(0.08 * i, 2) for i in range(hours)], # Rises from 37.1 to 38.9°C (fever)
    "SBP": [118 - int(i * 1.2) for i in range(hours)],         # Drops from 118 to 90 mmHg (hypotension)
    "MAP": [78 - int(i * 0.9) for i in range(hours)],          # Drops from 78 to 57 mmHg
    "DBP": [68 - int(i * 0.7) for i in range(hours)],
    "Resp": [18 + int(i * 0.5) for i in range(hours)],         # Rises from 18 to 29 bpm (tachypnea)
    "EtCO2": [np.nan] * hours,
    "Glucose": [130 + int(i * 3.5) if i % 4 == 0 else np.nan for i in range(hours)],
    "WBC": [11.2 + round(0.4 * i, 1) if i in (1, 6, 12, 18, 24) else np.nan for i in range(hours)], # Leukocytosis
    "Creatinine": [1.1 + round(0.06 * i, 2) if i in (1, 12, 24) else np.nan for i in range(hours)],
    "Platelets": [180 - int(3 * i) if i in (1, 12, 24) else np.nan for i in range(hours)],
    "Lactate": [1.8 + round(0.15 * i, 2) if i in (1, 6, 12, 18, 24) else np.nan for i in range(hours)], # Hyperlactatemia
    "BUN": [20 + int(0.8 * i) if i in (1, 12, 24) else np.nan for i in range(hours)],
    "Hct": [34.0 - round(0.2 * i, 1) if i in (1, 12) else np.nan for i in range(hours)],
    "pH": [7.38 - round(0.008 * i, 3) if i in (1, 6, 12, 18) else np.nan for i in range(hours)],
})
df_sepsis.to_csv(os.path.join(SAMPLE_DIR, "sample_positive_case.csv"), index=False)

# 3. Stable ICU case
df_negative = pd.DataFrame({
    "patient": ["p_healthy_303"] * hours,
    "ICULOS": list(range(1, hours + 1)),
    "Age": [45] * hours,
    "Gender": [1] * hours,
    "Unit1": [0.0] * hours,
    "Unit2": [1.0] * hours,
    "HospAdmTime": [-2.0] * hours,
    "HR": [70 + int(np.cos(i)*3) for i in range(hours)],
    "O2Sat": [99 for _ in range(hours)],
    "Temp": [36.6 + round(0.01 * np.sin(i), 2) for i in range(hours)],
    "SBP": [115 + int(np.sin(i)*2) for i in range(hours)],
    "MAP": [80 for _ in range(hours)],
    "DBP": [70 for _ in range(hours)],
    "Resp": [14 for _ in range(hours)],
    "EtCO2": [np.nan] * hours,
    "Glucose": [95 if i in (1, 12) else np.nan for i in range(hours)],
    "WBC": [6.5 if i in (1, 12) else np.nan for i in range(hours)],
    "Creatinine": [0.8 if i in (1, 12) else np.nan for i in range(hours)],
    "Platelets": [240 if i in (1, 12) else np.nan for i in range(hours)],
    "Lactate": [0.9 if i in (1, 12) else np.nan for i in range(hours)],
})
df_negative.to_csv(os.path.join(SAMPLE_DIR, "sample_negative_case.csv"), index=False)

# 4. Incompatible CSV
df_invalid = pd.DataFrame({
    "id": [1, 2, 3],
    "item_name": ["Product A", "Product B", "Product C"],
    "price": [19.99, 29.99, 49.99]
})
df_invalid.to_csv(os.path.join(SAMPLE_DIR, "sample_invalid_missing.csv"), index=False)
print("Sample fixtures generated successfully in sample_data/")

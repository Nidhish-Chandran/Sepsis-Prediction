"""
Prediction service for SepsisLens.
Loads the Random Forest artifact and produces clinical predictions.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from .config import (
    VITALS, LABS, STATIC, ALL_RAW_FEATURES,
    DEFAULT_THRESHOLD, FEATURE_DISPLAY_NAMES
)
from .feature_engineering import build_features

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class SepsisPredictor:
    """
    Inference service for SepsisLens Random Forest model.
    """
    _instance = None

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self):
        model_path = os.path.join(ARTIFACTS_DIR, "random_forest.joblib")
        preproc_path = os.path.join(ARTIFACTS_DIR, "preprocessing.joblib")
        meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(preproc_path):
            raise FileNotFoundError(
                f"Model artifacts missing in {ARTIFACTS_DIR}. Please run ml/train.py first."
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preproc_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    @property
    def expected_features(self) -> List[str]:
        return self.preprocessor.get("features", [])

    @property
    def decision_threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", DEFAULT_THRESHOLD))

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validates the incoming CSV dataframe.
        Returns:
            (is_valid, error_messages, summary_stats)
        """
        errors = []
        if df is None or df.empty:
            return False, ["The uploaded CSV file is empty."], {}

        if len(df) > 100_000:
            errors.append(f"File exceeds maximum allowed preview rows (found {len(df):,} rows).")

        # Check for clinical columns
        present_features = [c for c in ALL_RAW_FEATURES if c in df.columns]
        if len(present_features) < 3:
            return False, [
                "The CSV lacks sufficient clinical ICU columns. "
                f"Expected at least 3 clinical features among {ALL_RAW_FEATURES[:10]}... "
                f"Only found {present_features}."
            ], {}

        # Compute missingness across present columns
        missing_counts = df[present_features].isna().sum().to_dict()
        missing_pct = (df[present_features].isna().mean() * 100).round(1).to_dict()

        patient_col = "patient" if "patient" in df.columns else ("Patient_ID" if "Patient_ID" in df.columns else None)
        patient_count = df[patient_col].nunique() if patient_col else 1

        summary = {
            "total_rows": len(df),
            "patient_count": patient_count,
            "present_features_count": len(present_features),
            "missing_features_count": len(ALL_RAW_FEATURES) - len(present_features),
            "present_features": present_features,
            "missing_pct": missing_pct,
        }

        return True, errors, summary

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes feature engineering, imputation, and Random Forest inference.
        """
        thr = threshold if threshold is not None else self.decision_threshold

        # 1. Feature Engineering
        X_eng, y_true, meta = build_features(df)

        # 2. Reindex to training feature order
        features = self.expected_features
        X_aligned = X_eng.reindex(columns=features)

        # 3. Impute using fitted median values
        imputer = self.preprocessor["imputer"]
        X_imputed = pd.DataFrame(
            imputer.transform(X_aligned),
            columns=features,
            index=X_aligned.index
        ).astype("float32")

        # 4. Predict probabilities using Random Forest
        # Output is fraction of trees voting positive (uncalibrated probability score)
        scores = self.model.predict_proba(X_imputed)[:, 1]
        preds = (scores >= thr).astype(int)

        # 5. Build detailed hourly results
        records = []
        for i in range(len(meta)):
            pid = str(meta["patient"].iloc[i])
            hosp = str(meta["hospital"].iloc[i])
            iculos = int(meta["ICULOS"].iloc[i])
            score = float(scores[i])
            pred = int(preds[i])
            verdict = "SEPSIS ALERT" if pred == 1 else "Low Risk"
            verdict_badge = "danger" if pred == 1 else "success"

            rec = {
                "row_index": i,
                "patient": pid,
                "hospital": hosp,
                "iculos": iculos,
                "risk_score": round(score, 4),
                "risk_percent": round(score * 100, 1),
                "prediction": pred,
                "verdict": verdict,
                "badge": verdict_badge,
            }
            if "SepsisLabel" in meta.columns:
                rec["true_label"] = int(meta["SepsisLabel"].iloc[i])
            records.append(rec)

        results_df = pd.DataFrame(records)

        # 6. Patient-level aggregated summary
        patient_summaries = []
        for pid, group in results_df.groupby("patient", sort=False):
            max_score = float(group["risk_score"].max())
            latest_rec = group.iloc[-1]
            alarm_hours = group[group["prediction"] == 1]["iculos"].tolist()
            ever_alarm = len(alarm_hours) > 0

            summary = {
                "patient": pid,
                "total_hours": len(group),
                "latest_hour": int(latest_rec["iculos"]),
                "latest_risk_score": float(latest_rec["risk_score"]),
                "latest_risk_percent": float(latest_rec["risk_percent"]),
                "latest_verdict": str(latest_rec["verdict"]),
                "max_risk_score": round(max_score, 4),
                "max_risk_percent": round(max_score * 100, 1),
                "ever_sepsis_alarm": ever_alarm,
                "first_alarm_hour": alarm_hours[0] if ever_alarm else None,
                "total_alarm_hours": len(alarm_hours),
            }
            patient_summaries.append(summary)

        return {
            "records": records,
            "patient_summaries": patient_summaries,
            "threshold_used": thr,
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "total_records": len(records),
            "positive_alarms": int((preds == 1).sum()),
            "alarm_rate": round(float((preds == 1).mean() * 100), 2),
            "X_imputed": X_imputed,  # retained for explainability calculations
        }

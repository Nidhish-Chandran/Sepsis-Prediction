"""
Evaluation utilities and comparison data from ML_Project.ipynb.
"""
import os
import json
from typing import Dict, Any

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def get_model_comparison_data() -> Dict[str, Any]:
    """
    Returns recorded evaluation results from the original ML_Project.ipynb notebook.
    """
    path = os.path.join(ARTIFACTS_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_early_warning_data() -> Dict[str, Any]:
    """
    Returns early warning lead-time analysis recorded in Cell 26 and Cell 27.
    """
    return {
        "models": {
            "Random Forest": {
                "Septic_patients": 48,
                "Detected": 24,
                "Detection_rate": 0.500,
                "Median_lead_h": 27.5,
                "Mean_lead_h": 67.625,
                "False_alarm_rate": 0.215
            }
        },
        "consecutive_hour_sweep": [
            {"consecutive_h": 1, "detection_rate": 0.500, "median_lead_h": 27.5, "false_alarm_rate": 0.215},
            {"consecutive_h": 2, "detection_rate": 0.271, "median_lead_h": 38.0, "false_alarm_rate": 0.019},
            {"consecutive_h": 3, "detection_rate": 0.271, "median_lead_h": 34.0, "false_alarm_rate": 0.009},
            {"consecutive_h": 4, "detection_rate": 0.250, "median_lead_h": 34.5, "false_alarm_rate": 0.007},
            {"consecutive_h": 6, "detection_rate": 0.208, "median_lead_h": 32.5, "false_alarm_rate": 0.003},
        ],
        "threshold_sweep": [
            {"alerted_pct": 50.0, "threshold": 0.0493, "precision": 0.0262, "recall": 0.8600, "f1": 0.0508},
            {"alerted_pct": 20.0, "threshold": 0.1065, "precision": 0.0494, "recall": 0.6489, "f1": 0.0918},
            {"alerted_pct": 10.0, "threshold": 0.1708, "precision": 0.0646, "recall": 0.4244, "f1": 0.1122},
            {"alerted_pct": 5.0,  "threshold": 0.2631, "precision": 0.0771, "recall": 0.2533, "f1": 0.1183},
            {"alerted_pct": 3.0,  "threshold": 0.3609, "precision": 0.1003, "recall": 0.1978, "f1": 0.1331},
            {"alerted_pct": 2.0,  "threshold": 0.4484, "precision": 0.0811, "recall": 0.1067, "f1": 0.0921},
            {"alerted_pct": 1.0,  "threshold": 0.5353, "precision": 0.0642, "recall": 0.0422, "f1": 0.0509},
            {"alerted_pct": 0.5,  "threshold": 0.5893, "precision": 0.0676, "recall": 0.0222, "f1": 0.0334},
        ]
    }

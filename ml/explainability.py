"""
Explainability module for SepsisLens.
Provides SHAP local explanations and global feature importance.
"""
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional
from .config import FEATURE_DISPLAY_NAMES, KEY, VITALS, LABS, STATIC
from .predictor import SepsisPredictor

class SepsisExplainer:
    """
    Explainability engine using SHAP TreeExplainer and Random Forest feature importances.
    """
    def __init__(self, predictor: Optional[SepsisPredictor] = None):
        self.predictor = predictor or SepsisPredictor.get_instance()
        self._tree_explainer = None

    @property
    def tree_explainer(self):
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self.predictor.model)
        return self._tree_explainer

    def get_global_importance(self, top_n: int = 25) -> List[Dict[str, Any]]:
        """
        Returns top global features ranked by Gini feature importance.
        """
        rf = self.predictor.model
        features = self.predictor.expected_features
        importances = rf.feature_importances_

        indices = np.argsort(importances)[::-1][:top_n]
        results = []
        for rank, idx in enumerate(indices, start=1):
            feat_name = features[idx]
            base_signal = feat_name.split("_")[0]
            readable = FEATURE_DISPLAY_NAMES.get(base_signal, feat_name)
            
            # Identify feature category
            if "_msg" in feat_name:
                category = "Missingness Indicator"
            elif "_cnt" in feat_name:
                category = "Sampling Frequency"
            elif "_mean" in feat_name or "_min" in feat_name or "_max" in feat_name or "_std" in feat_name:
                category = "6h Rolling Trend"
            elif "_d1" in feat_name:
                category = "1h Rate of Change"
            elif "_dbase" in feat_name:
                category = "Drift from Baseline"
            elif base_signal in STATIC:
                category = "Demographic / Static"
            elif base_signal in VITALS:
                category = "Vital Sign"
            elif base_signal in LABS:
                category = "Laboratory Test"
            else:
                category = "Clinical Feature"

            results.append({
                "rank": rank,
                "feature": feat_name,
                "display_name": f"{readable} ({feat_name})",
                "importance": round(float(importances[idx]), 5),
                "importance_percent": round(float(importances[idx] * 100), 2),
                "category": category,
            })
        return results

    def explain_row(
        self,
        X_row: pd.DataFrame,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Computes local SHAP contributions for a single imputed row.
        """
        explainer = self.tree_explainer
        features = self.predictor.expected_features
        X_aligned = X_row.reindex(columns=features)

        shap_vals = explainer.shap_values(X_aligned)
        
        # Handle SHAP output shapes
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]  # Positive class (Sepsis)
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        row_vals = X_aligned.iloc[0].values

        # Sort contributions
        pos_indices = [i for i in np.argsort(sv)[::-1] if sv[i] > 0][:top_k]
        neg_indices = [i for i in np.argsort(sv) if sv[i] < 0][:top_k]

        risk_increasing = []
        for idx in pos_indices:
            feat = features[idx]
            base = feat.split("_")[0]
            risk_increasing.append({
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(base, feat),
                "value": round(float(row_vals[idx]), 3),
                "shap_value": round(float(sv[idx]), 4),
                "impact": "Increases Risk",
            })

        risk_decreasing = []
        for idx in neg_indices:
            feat = features[idx]
            base = feat.split("_")[0]
            risk_decreasing.append({
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(base, feat),
                "value": round(float(row_vals[idx]), 3),
                "shap_value": round(float(sv[idx]), 4),
                "impact": "Decreases Risk",
            })

        # Top overall absolute features for waterfall/bar chart
        all_indices = np.argsort(np.abs(sv))[::-1][:top_k]
        chart_features = []
        for idx in all_indices:
            feat = features[idx]
            base = feat.split("_")[0]
            chart_features.append({
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(base, feat),
                "value": round(float(row_vals[idx]), 3),
                "shap_value": round(float(sv[idx]), 4),
                "is_positive": bool(sv[idx] > 0),
            })

        base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

        return {
            "base_value": round(base_value, 4),
            "risk_increasing": risk_increasing,
            "risk_decreasing": risk_decreasing,
            "chart_features": chart_features,
        }

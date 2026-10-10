"""
Preprocessor module for SepsisLens.
Encapsulates missing value imputation and feature ordering.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from typing import List, Optional

class SepsisFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Standard preprocessor for SepsisLens:
    1. Ensures all expected feature columns are present in exact order.
    2. Imputes missing values with median values learned during training.
    """
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, X: pd.DataFrame, y=None):
        if self.feature_names is None or len(self.feature_names) == 0:
            self.feature_names = list(X.columns)
        X_aligned = X.reindex(columns=self.feature_names)
        self.imputer.fit(X_aligned)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names:
            X_aligned = X.reindex(columns=self.feature_names)
        else:
            X_aligned = X.copy()
        
        imputed_array = self.imputer.transform(X_aligned)
        return pd.DataFrame(imputed_array, columns=self.feature_names, index=X.index).astype("float32")

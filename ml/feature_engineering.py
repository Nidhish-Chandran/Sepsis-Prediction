"""
Feature engineering module for SepsisLens.
Reproduces the 164-feature pipeline from ML_Project.ipynb (Cell 7).
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .config import VITALS, LABS, STATIC, KEY, ROLL_WINDOW

def build_features(df: pd.DataFrame, roll: int = ROLL_WINDOW) -> Tuple[pd.DataFrame, Optional[np.ndarray], pd.DataFrame]:
    """
    Constructs the exact 164 features from ICU time-series data as designed in the notebook.
    
    Returns:
        X: DataFrame of 164 engineered features
        y: 1D array of SepsisLabel (or None if not present in input)
        meta: DataFrame of metadata columns (patient, hospital, ICULOS, SepsisLabel)
    """
    df = df.copy()

    # Normalize patient ID column name
    if "patient" not in df.columns and "Patient_ID" in df.columns:
        df["patient"] = df["Patient_ID"].astype(str)
    elif "patient" not in df.columns:
        df["patient"] = "PATIENT_001"

    # Normalize hospital column name if missing
    if "hospital" not in df.columns:
        df["hospital"] = "A"

    # Ensure ICULOS is present and numeric
    if "ICULOS" not in df.columns or df["ICULOS"].isna().all():
        if "Hour" in df.columns:
            df["ICULOS"] = df["Hour"] + 1
        else:
            df["ICULOS"] = df.groupby("patient").cumcount() + 1
    else:
        df["ICULOS"] = pd.to_numeric(df["ICULOS"], errors="coerce").fillna(1)

    # Ensure all STATIC, VITALS, and LABS columns exist in df
    for col in STATIC + VITALS + LABS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort so each patient's hours are in chronological order
    df = df.sort_values(["patient", "ICULOS"]).reset_index(drop=True)
    pid = df["patient"]

    # 1. Static features (6)
    parts = [df[STATIC].astype("float32")]

    # 2. Missingness indicators for LABS (26)
    m = df[LABS].isna().astype("int8")
    m.columns = [f"{c}_msg" for c in LABS]
    parts.append(m)

    # 3. Cumulative measurement counts for KEY signals (14)
    cnt = df[KEY].notna().astype("int16").groupby(pid, observed=True).cumsum()
    cnt.columns = [f"{c}_cnt" for c in KEY]
    parts.append(cnt.astype("int16"))

    # 4. Forward-filled vitals and labs (34)
    filled = df[VITALS + LABS].groupby(pid, observed=True).ffill().astype("float32")
    parts.append(filled)

    # 5. Rolling statistics over the last `roll` hours (14 * 4 = 56)
    g = filled[KEY].groupby(pid, observed=True, sort=False)
    r = g.rolling(roll, min_periods=1)
    for name in ["mean", "min", "max", "std"]:
        out = getattr(r, name)().reset_index(level=0, drop=True).sort_index()
        out.columns = [f"{c}_{name}{roll}" for c in KEY]
        parts.append(out.astype("float32"))

    # 6. Hour-to-hour change (d1) (14)
    d1 = g.diff()
    d1.columns = [f"{c}_d1" for c in KEY]
    parts.append(d1.astype("float32"))

    # 7. Drift from baseline (first reading) (14)
    base = g.transform("first")
    delta = pd.DataFrame(
        (filled[KEY].values - base.values).astype("float32"),
        columns=[f"{c}_dbase" for c in KEY],
        index=filled.index
    )
    parts.append(delta)

    # Concatenate all parts
    X = pd.concat(parts, axis=1)
    X.columns = [str(c) for c in X.columns]

    y = None
    if "SepsisLabel" in df.columns:
        y = pd.to_numeric(df["SepsisLabel"], errors="coerce").fillna(0).astype("int8").values

    meta_cols = ["patient", "hospital", "ICULOS"]
    if "SepsisLabel" in df.columns:
        meta_cols.append("SepsisLabel")
    meta = df[meta_cols].copy()

    return X, y, meta

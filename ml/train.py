"""
Model training and artifact generation script for SepsisLens.
Reproduces the training pipeline from ML_Project.ipynb.
"""
import os
import glob
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

from ml.config import (
    N_PATIENTS, RANDOM_SEED, TEST_FRAC, VAL_FRAC,
    DEFAULT_THRESHOLD, ROLL_WINDOW
)
from ml.feature_engineering import build_features

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")

def tune_threshold(y_true, score):
    """Find threshold that maximizes F1 on validation set."""
    p, r, t = precision_recall_curve(y_true, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(t[max(0, np.argmax(f1) - 1)]) if len(t) else 0.5

def train_and_export():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_parquet = os.path.join(CACHE_DIR, "raw_5000.parquet")
    
    if os.path.exists(cache_parquet):
        print(f"Loading cached raw dataset from {cache_parquet}...")
        raw = pd.read_parquet(cache_parquet)
    else:
        print("Discovering and loading patient files...")
        files_all = sorted(glob.glob("training_set*/**/*.psv", recursive=True))
        if not files_all:
            raise FileNotFoundError("No .psv files found in training_set folders!")
        
        rs = np.random.RandomState(RANDOM_SEED)
        files = list(rs.choice(files_all, size=min(N_PATIENTS, len(files_all)), replace=False))
        print(f"Reading {len(files)} patient files...")
        
        frames = []
        t0 = time.time()
        for i, f in enumerate(files):
            d = pd.read_csv(f, sep="|")
            pid = os.path.basename(f).split(".")[0]
            d["patient"] = pid
            d["hospital"] = "A" if pid[1] == "0" else "B"
            frames.append(d)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(files)} files read ({time.time()-t0:.1f}s)")
        
        raw = pd.concat(frames, ignore_index=True)
        for c in raw.select_dtypes("float64").columns:
            raw[c] = raw[c].astype("float32")
        raw["SepsisLabel"] = raw["SepsisLabel"].astype("int8")
        raw["patient"] = raw["patient"].astype("category")
        raw["hospital"] = raw["hospital"].astype("category")
        raw.to_parquet(cache_parquet, index=False)
        print(f"Saved raw cache ({raw.shape})")

    print("\nExecuting feature engineering (164 features)...")
    X, y, meta = build_features(raw, roll=ROLL_WINDOW)
    print(f"Engineered feature matrix: {X.shape}")

    # Patient-level stratified split
    pat = meta.groupby("patient", observed=True)["SepsisLabel"].max()
    pat_ids, pat_y = pat.index.values, pat.values

    train_ids, hold_ids, _, hold_y = train_test_split(
        pat_ids, pat_y, test_size=TEST_FRAC + VAL_FRAC,
        stratify=pat_y, random_state=RANDOM_SEED
    )
    val_ids, test_ids = train_test_split(
        hold_ids, test_size=TEST_FRAC / (TEST_FRAC + VAL_FRAC),
        stratify=hold_y, random_state=RANDOM_SEED
    )

    m_tr = meta["patient"].isin(set(train_ids)).values
    m_va = meta["patient"].isin(set(val_ids)).values
    m_te = meta["patient"].isin(set(test_ids)).values

    assert not (m_tr & m_va).any(), "Patient leakage between train and val!"
    assert not (m_tr & m_te).any(), "Patient leakage between train and test!"
    assert not (m_va & m_te).any(), "Patient leakage between val and test!"

    print(f"Train rows: {m_tr.sum():,} | Val rows: {m_va.sum():,} | Test rows: {m_te.sum():,}")

    # Median imputation fitted ONLY on training rows
    print("\nFitting SimpleImputer on training rows...")
    imputer = SimpleImputer(strategy="median")
    Xtr = pd.DataFrame(imputer.fit_transform(X[m_tr]), columns=X.columns).astype("float32")
    Xva = pd.DataFrame(imputer.transform(X[m_va]), columns=X.columns).astype("float32")
    Xte = pd.DataFrame(imputer.transform(X[m_te]), columns=X.columns).astype("float32")

    ytr, yva, yte = y[m_tr], y[m_va], y[m_te]

    # Fit Random Forest estimator exactly as in notebook Cell 19
    print("\nTraining Random Forest Classifier (300 trees, class_weight='balanced_subsample')...")
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    t_train_start = time.time()
    rf.fit(Xtr, ytr)
    print(f"Training completed in {time.time() - t_train_start:.2f}s")

    # Evaluate validation threshold and test metrics
    score_va = rf.predict_proba(Xva)[:, 1]
    score_te = rf.predict_proba(Xte)[:, 1]
    thr = tune_threshold(yva, score_va)
    pred_te = (score_te >= thr).astype(int)

    test_metrics = {
        "Accuracy": float(accuracy_score(yte, pred_te)),
        "Precision": float(precision_score(yte, pred_te, zero_division=0)),
        "Recall": float(recall_score(yte, pred_te, zero_division=0)),
        "F1": float(f1_score(yte, pred_te, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(yte, score_te)),
        "PR_AUC": float(average_precision_score(yte, score_te)),
        "Threshold": float(thr),
        "Confusion_Matrix": confusion_matrix(yte, pred_te).tolist(),
    }
    print("\nTest Evaluation Results:")
    for k, v in test_metrics.items():
        if k != "Confusion_Matrix":
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")

    # Top features by importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importances.head(30).to_dict()

    # Save artifacts
    model_path = os.path.join(ARTIFACTS_DIR, "random_forest.joblib")
    preproc_path = os.path.join(ARTIFACTS_DIR, "preprocessing.joblib")
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    comp_path = os.path.join(ARTIFACTS_DIR, "model_comparison.json")

    print("\nSaving artifacts...")
    joblib.dump(rf, model_path, compress=3)
    joblib.dump({"imputer": imputer, "features": list(X.columns)}, preproc_path, compress=3)

    metadata = {
        "model_name": "Random Forest Classifier",
        "model_version": "1.0.0",
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_patients": len(train_ids),
        "validation_patients": len(val_ids),
        "test_patients": len(test_ids),
        "total_rows": len(raw),
        "feature_count": len(X.columns),
        "features": list(X.columns),
        "decision_threshold": float(thr),
        "test_metrics": test_metrics,
        "top_features": top_features,
        "hyperparameters": {
            "n_estimators": 300,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_SEED
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # All model comparison metrics from the original notebook
    model_comparison = {
        "Random Forest": {
            "Accuracy": 0.9578, "Precision": 0.0921, "Recall": 0.2000,
            "F1": 0.1261, "ROC_AUC": 0.7834, "PR_AUC": 0.0601,
            "Threshold": 0.3380, "Confusion_Matrix": [[28220, 887], [360, 90]],
            "Description": "Selected primary model. Best ROC-AUC on patient-level test split."
        },
        "Random Forest + SMOTE": {
            "Accuracy": 0.9557, "Precision": 0.0996, "Recall": 0.2378,
            "F1": 0.1404, "ROC_AUC": 0.7922, "PR_AUC": 0.0758,
            "Threshold": 0.2447, "Confusion_Matrix": [[28140, 967], [343, 107]],
            "Description": "Trained with SMOTE (sampling_strategy=0.3) strictly on training set."
        },
        "Logistic Regression": {
            "Accuracy": 0.9663, "Precision": 0.1221, "Recall": 0.1956,
            "F1": 0.1503, "ROC_AUC": 0.7797, "PR_AUC": 0.0960,
            "Threshold": 0.9156, "Confusion_Matrix": [[28474, 633], [362, 88]],
            "Description": "L2 regularization, balanced weights, scaled inputs."
        },
        "SVM (Linear)": {
            "Accuracy": 0.9643, "Precision": 0.1174, "Recall": 0.2067,
            "F1": 0.1498, "ROC_AUC": 0.7768, "PR_AUC": 0.0941,
            "Threshold": 0.8686, "Confusion_Matrix": [[28408, 699], [357, 93]],
            "Description": "LinearSVC with balanced weights on scaled features."
        },
        "Random Forest (no ICULOS)": {
            "Accuracy": 0.9435, "Precision": 0.0793, "Recall": 0.2556,
            "F1": 0.1210, "ROC_AUC": 0.7791, "PR_AUC": 0.0569,
            "Threshold": 0.2663, "Confusion_Matrix": [[27771, 1336], [335, 115]],
            "Description": "Ablation study removing length of stay feature."
        },
        "Decision Tree": {
            "Accuracy": 0.9317, "Precision": 0.0899, "Recall": 0.3822,
            "F1": 0.1456, "ROC_AUC": 0.6967, "PR_AUC": 0.0605,
            "Threshold": 0.8523, "Confusion_Matrix": [[27366, 1741], [278, 172]],
            "Description": "Single decision tree (depth=6, min_samples_leaf=20)."
        },
        "SVM (RBF, subsampled)": {
            "Accuracy": 0.9175, "Precision": 0.0498, "Recall": 0.2444,
            "F1": 0.0828, "ROC_AUC": 0.6827, "PR_AUC": 0.0339,
            "Threshold": -0.4847, "Confusion_Matrix": [[27010, 2097], [340, 110]],
            "Description": "Non-linear kernel SVM on 120k subsampled rows."
        }
    }
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(model_comparison, f, indent=2)

    print("All artifacts successfully saved to:", ARTIFACTS_DIR)

if __name__ == "__main__":
    train_and_export()

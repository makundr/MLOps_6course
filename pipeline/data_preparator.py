"""
Этап 3: Подготовка данных
"""
import os, pickle
import pandas as pd
import numpy as np
import yaml

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


DROP_COLS = ["INSR_BEGIN", "INSR_END", "OBJECT_ID", "CLAIM_PAID"]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute",  SimpleImputer(strategy="most_frequent")),
        ("encode",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def run(df, batch_idx, existing_preprocessor=None):
    cfg = load_config()
    target = cfg["data"]["target_column"]
    test_size = cfg["preparation"]["test_size"]

    if target not in df.columns:
        raise ValueError(f"Целевая колонка '{target}' не найдена")

    y = df[target].values
    X = df.drop(columns=[target] + [c for c in DROP_COLS if c in df.columns],
                errors="ignore")

    split = int(len(X) * (1 - test_size))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    num_cols = X_tr.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_tr.columns
                if c not in num_cols and X_tr[c].nunique() <= 20]

    print(f"Числовых: {len(num_cols)}, категориальных: {len(cat_cols)}")

    if existing_preprocessor is not None:
        preprocessor = existing_preprocessor
        X_tr_proc = preprocessor.transform(X_tr)
        X_te_proc = preprocessor.transform(X_te)
        print(f"  Используем существующий препроцессор")
    else:
        preprocessor = build_preprocessor(num_cols, cat_cols)
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_te_proc = preprocessor.transform(X_te)

    print(f"  train={X_tr_proc.shape}, test={X_te_proc.shape}")

    os.makedirs("models", exist_ok=True)
    prep_path = f"models/preprocessor_batch_{batch_idx:04d}.pkl"
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)

    return X_tr_proc, X_te_proc, y_tr, y_te, preprocessor


def load_latest_preprocessor():
    import glob
    files = sorted(glob.glob("models/preprocessor_batch_*.pkl"))
    if not files:
        return None
    with open(files[-1], "rb") as f:
        return pickle.load(f)

"""
Этап 6: Обслуживание модели
"""
import os, pickle, time, json
import pandas as pd
from datetime import datetime


PROD_PATH = "models/production_model.pkl"

DROP_COLS = ["INSR_BEGIN", "INSR_END", "OBJECT_ID", "CLAIM_PAID"]


def save_production(model, preprocessor, model_name, batch_idx):
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "batch_index": batch_idx,
        "saved_at": datetime.now().isoformat(),
    }
    with open(PROD_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Продовая модель сохранена: {model_name} (батч №{batch_idx})")


def predict(df, target_col=None):
    if not os.path.exists(PROD_PATH):
        raise RuntimeError("Продовая модель не найдена. Запустите update.")

    with open(PROD_PATH, "rb") as f:
        bundle = pickle.load(f)

    preprocessor = bundle["preprocessor"]
    model        = bundle["model"]

    X = df.drop(columns=[target_col] + DROP_COLS, errors="ignore") if target_col else \
        df.drop(columns=DROP_COLS, errors="ignore")

    if "HAS_CLAIM" in X.columns:
        X = X.drop(columns=["HAS_CLAIM"])

    t0 = time.time()
    X_proc = preprocessor.transform(X)
    preds  = model.predict(X_proc)
    ms = (time.time() - t0) * 1000

    print(f"  Прогноз: {len(preds)} строк за {ms:.1f} мс")

    # Лог производительности
    os.makedirs("logs", exist_ok=True)
    with open("logs/performance.jsonl", "a") as f:
        f.write(json.dumps({
            "date": datetime.now().isoformat(),
            "n_rows": len(preds),
            "ms": round(ms, 1),
        }) + "\n")

    result = df.copy()
    result["predict"] = preds
    return result

"""
Этап 5: Валидация
"""
import os, json
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


HISTORY_FILE = "models/metrics_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def validate(models, X_test, y_test, batch_idx):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_claim    = float(f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0))

        auc = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, proba))
            except Exception:
                pass

        results[name] = {"accuracy": acc, "f1_weighted": f1_weighted, "f1_claim": f1_claim, "auc": auc}
        auc_str = f"  auc={auc:.3f}" if auc else ""
        print(f"  {name}: acc={acc:.3f}  f1_weighted={f1_weighted:.3f}  f1_claim={f1_claim:.3f}")

    best_name = max(results, key=lambda n: results[n]["f1_claim"])
    print(f"  Лучшая модель: {best_name}")

    history = load_history()
    history.append({
        "batch_index": batch_idx,
        "date": datetime.now().isoformat(),
        "best_model": best_name,
        "metrics": results,
    })
    os.makedirs("models", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return best_name, models[best_name], results

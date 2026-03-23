"""
Этап 2: Анализ данных
"""
import os, json
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND = True
except ImportError:
    MLXTEND = False


def check_quality(df, batch_idx):
    report = {
        "batch_index": batch_idx,
        "date": datetime.now().isoformat(),
        "n_rows": len(df),
        "missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "duplicates": int(df.duplicated().sum()),
        "missing_per_col": {c: round(v * 100, 1) for c, v in df.isnull().mean().items() if v > 0}
    }

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/quality_batch_{batch_idx:04d}.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Качество: пропуски={report['missing_pct']}%, "
          f"дубликаты={report['duplicates']}")
    return report


def find_association_rules(df, batch_idx):
    if not MLXTEND:
        print("  mlxtend не установлен, правила пропущены")
        return

    cat_cols = [c for c in ["SEX", "INSR_TYPE", "TYPE_VEHICLE", "USAGE", "HAS_CLAIM"]
                if c in df.columns]
    if len(cat_cols) < 2:
        return

    sample = df[cat_cols].dropna().sample(min(5000, len(df)), random_state=42)
    dummies = pd.get_dummies(sample.astype(str)).astype(bool)

    try:
        freq = apriori(dummies, min_support=0.05, use_colnames=True)
        if freq.empty:
            print("  Частые наборы не найдены")
            return
        rules = association_rules(freq, metric="confidence", min_threshold=0.5)
        rules = rules[rules["lift"] > 1.0].sort_values("lift", ascending=False).head(10)

        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules.to_csv(f"reports/rules_batch_{batch_idx:04d}.csv", index=False)

        print(f"  Найдено {len(rules)} ассоциативных правил")
        for _, r in rules.head(3).iterrows():
            print(f"    [{r['antecedents']}] → [{r['consequents']}]  "
                  f"conf={r['confidence']:.2f} lift={r['lift']:.2f}")
    except Exception as e:
        print(f"  Ошибка поиска правил: {e}")


def clean(df, quality_report):
    before = len(df)

    df = df.drop_duplicates()

    # Удаляем колонки с критически большим числом пропусков
    bad_cols = [c for c, v in quality_report["missing_per_col"].items() if v > 70]
    bad_cols = [c for c in bad_cols if c != "CLAIM_PAID"]
    if bad_cols:
        df = df.drop(columns=bad_cols, errors="ignore")
        print(f"  Удалены колонки: {bad_cols}")

    for col in ["PREMIUM", "INSURED_VALUE"]:
        if col in df.columns:
            df = df[~(df[col] < 0)]

    print(f"  Очистка: {before} → {len(df)} строк")
    return df.reset_index(drop=True)


def run(df, batch_idx):
    print(f"[Этап 2] Анализ батча №{batch_idx}")
    quality = check_quality(df, batch_idx)
    find_association_rules(df, batch_idx)
    df_clean = clean(df, quality)
    return df_clean

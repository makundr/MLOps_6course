
import argparse, sys, os, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_collector    import load_config, next_batch, reset_state
from pipeline.data_analyzer     import run as analyze
from pipeline.data_preparator   import run as prepare, load_latest_preprocessor
from pipeline.trainer           import train
from pipeline.validator         import validate
from pipeline.server            import save_production, predict


def mode_update():
    print("=== UPDATE ===")
    cfg = load_config()

    batch, batch_idx = next_batch(cfg)
    if batch is None:
        print("Данные закончились. Используйте: python run.py -mode reset")
        return False

    clean = analyze(batch, batch_idx)

    existing_prep = load_latest_preprocessor() if batch_idx > 0 else None
    X_train, X_test, y_train, y_test, preprocessor = prepare(
        clean, batch_idx, existing_prep
    )

    models = train(X_train, y_train, batch_idx)

    best_name, best_model, metrics = validate(models, X_test, y_test, batch_idx)

    save_production(best_model, preprocessor, best_name, batch_idx)

    print(f"=== Батч №{batch_idx} завершён ===")
    return True


def mode_inference(filepath):
    cfg = load_config()
    target = cfg["data"]["target_column"]
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Загружено: {len(df)} строк")

    result = predict(df, target_col=target)

    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"reports/inference_{ts}.csv"
    result.to_csv(out, index=False)
    print(f"  Результат: {out}")
    return out


def mode_summary():
    lines = [
        "=" * 60,
        "ОТЧЁТ МОНИТОРИНГА",
        f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]

    # Качество данных
    quality_files = sorted(Path("reports").glob("quality_batch_*.json"))
    lines.append(f"\nБатчей обработано: {len(quality_files)}")
    for qf in quality_files:
        with open(qf) as f:
            q = json.load(f)
        lines.append(
            f"  Батч №{q['batch_index']}: "
            f"строк={q['n_rows']}, "
            f"пропуски={q['missing_pct']}%, "
            f"дубликаты={q['duplicates']}"
        )

    # Метрики моделей
    if os.path.exists("models/metrics_history.json"):
        with open("models/metrics_history.json") as f:
            history = json.load(f)
        lines.append("\nМетрики моделей:")
        for entry in history:
            lines.append(f"  Батч №{entry['batch_index']} | лучшая: {entry['best_model']}")
            for name, m in entry["metrics"].items():
                auc = f"  auc={m['auc']:.3f}" if m.get("auc") else ""
                lines.append(f"    {name}: acc={m['accuracy']:.3f}  f1_claim={m['f1_claim']:.3f}{auc}")

    # Производительность
    if os.path.exists("logs/performance.jsonl"):
        with open("logs/performance.jsonl") as f:
            records = [json.loads(l) for l in f if l.strip()]
        if records:
            lines.append("\nПоследние вызовы inference:")
            for r in records[-5:]:
                lines.append(f"  {r['date'][:19]}  n={r['n_rows']}  {r['ms']} мс")

    lines.append("=" * 60)
    report = "\n".join(lines)

    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"reports/summary_{ts}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", required=True,
                        choices=["update", "inference", "summary", "reset"])
    parser.add_argument("-file", default=None)
    args = parser.parse_args()

    if args.mode == "update":
        result = mode_update()
        print(result)
        sys.exit(0 if result else 1)

    elif args.mode == "inference":
        if not args.file:
            print("Укажите файл")
            sys.exit(1)
        out = mode_inference(args.file)
        print(out)

    elif args.mode == "summary":
        mode_summary()

    elif args.mode == "reset":
        reset_state(load_config())

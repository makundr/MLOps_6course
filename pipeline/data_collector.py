"""
Этап 1: Сбор данных
"""
import os, json
import pandas as pd
import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_full_dataset(cfg):
    frames = []
    for path in cfg["data"]["source_files"]:
        df = pd.read_csv(path, sep=cfg["data"]["csv_separator"], low_memory=False)
        frames.append(df)
        print(f"  Загружен {path}: {df.shape}")
    data = pd.concat(frames, ignore_index=True)

    if "CLAIM_PAID" in data.columns:
        data["HAS_CLAIM"] = (data["CLAIM_PAID"].notna() & (data["CLAIM_PAID"] > 0)).astype(int)

    time_col = cfg["data"]["time_column"]
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
        data = data.sort_values(time_col).reset_index(drop=True)

    print(f"  Итого: {data.shape}, HAS_CLAIM: {data['HAS_CLAIM'].value_counts().to_dict()}")
    return data


def get_state(cfg):
    path = cfg["collection"]["state_file"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"current_batch": 0}


def save_state(cfg, state):
    with open(cfg["collection"]["state_file"], "w") as f:
        json.dump(state, f)


def reset_state(cfg):
    save_state(cfg, {"current_batch": 0})
    print("Состояние сброшено.")


def next_batch(cfg):
    data = load_full_dataset(cfg)
    state = get_state(cfg)

    batch_size = cfg["collection"]["batch_size"]
    idx = state["current_batch"]
    total = (len(data) + batch_size - 1) // batch_size

    if idx >= total:
        print("Все батчи уже обработаны. Используйте reset.")
        return None, None

    start = idx * batch_size
    end   = min(start + batch_size, len(data))
    batch = data.iloc[start:end].copy()

    raw_dir = cfg["collection"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"batch_{idx:04d}.csv")
    batch.to_csv(path, index=False)

    state["current_batch"] = idx + 1
    save_state(cfg, state)

    print(f"  Батч {idx}/{total-1}: строк={len(batch)}, пропусков={batch.isnull().sum().sum()}")
    return batch, idx

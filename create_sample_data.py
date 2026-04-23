"""
Генерация синтетических данных для тестирования CI/CD.
Создаёт небольшой датасет с той же структурой, что и реальные данные.
Запускается автоматически в GitHub Actions при отсутствии реальных данных.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 500

rng = np.random.default_rng(SEED)

dates_start = pd.date_range("2012-01-01", "2017-12-31", periods=N)
dates_end   = dates_start + pd.to_timedelta(rng.integers(180, 365, N), unit="D")

df = pd.DataFrame({
    "SEX":               rng.choice(["M", "F"], N),
    "INSR_BEGIN":        dates_start.strftime("%Y-%m-%d"),
    "INSR_END":          dates_end.strftime("%Y-%m-%d"),
    "EFFECTIVE_YR":      rng.integers(2011, 2018, N),
    "INSR_TYPE":         rng.choice(["Comprehensive", "Third Party", "Fire & Theft"], N),
    "INSURED_VALUE":     rng.uniform(50_000, 500_000, N).round(2),
    "PREMIUM":           rng.uniform(1_000, 20_000, N).round(2),
    "OBJECT_ID":         rng.integers(100_000, 999_999, N),
    "PROD_YEAR":         rng.integers(2000, 2018, N),
    "SEATS_NUM":         rng.choice([2, 4, 5, 7, 8], N),
    "CARRYING_CAPACITY": rng.uniform(0, 5, N).round(1),
    "TYPE_VEHICLE":      rng.choice(["Sedan", "SUV", "Truck", "Bus", "Motorcycle"], N),
    "CCM_TON":           rng.uniform(1000, 4000, N).round(0),
    "MAKE":              rng.choice(["Toyota", "Hyundai", "Isuzu", "Nissan", "Ford"], N),
    "USAGE":             rng.choice(["Private", "Commercial", "Government"], N),
    "CLAIM_PAID":        np.where(rng.random(N) < 0.15,
                                  rng.uniform(5_000, 100_000, N).round(2),
                                  np.nan),
})

os.makedirs("data/source", exist_ok=True)
out = "data/source/motor_data11-14lats.csv"
df.to_csv(out, index=False)
print(f"Сгенерировано {N} строк → {out}")

# MLOps MVP — Vehicle Insurance (Ethiopian Insurance Corporation)

Колонки: `SEX, INSR_BEGIN, INSR_END, EFFECTIVE_YR, INSR_TYPE, INSURED_VALUE,
PREMIUM, OBJECT_ID, PROD_YEAR, SEATS_NUM, CARRYING_CAPACITY, TYPE_VEHICLE,
CCM_TON, MAKE, USAGE, CLAIM_PAID`

Целевая переменная `HAS_CLAIM` создаётся автоматически из `CLAIM_PAID`:
- 1 — была страховая выплата
- 0 — выплаты не было

## Запуск

```bash
pip install -r requirements.txt

# Обработать следующий батч (5000 строк)
python run.py -mode update

# Применить модель к новым данным
python run.py -mode inference -file ./data/source/motor_data11-14lats.csv

# Отчёт о качестве и метриках
python run.py -mode summary

# Начать поток заново
python run.py -mode reset
```
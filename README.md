# Vehicle Insurance (Ethiopian Insurance Corporation)

Система инкрементального обучения для предсказания страховых выплат по автомобильным полисам. Обрабатывает данные батчами, обучает две модели (Decision Tree + MLP) и выбирает лучшую по F1 на тестовой выборке.

📁 **[Документация проекта](doc/)**
- [Описание задачи и пайплайна](doc/task.md)
- [Ожидаемые баллы](doc/grade.md)

---

## Колонки датасета

`SEX, INSR_BEGIN, INSR_END, EFFECTIVE_YR, INSR_TYPE, INSURED_VALUE,
PREMIUM, OBJECT_ID, PROD_YEAR, SEATS_NUM, CARRYING_CAPACITY, TYPE_VEHICLE,
CCM_TON, MAKE, USAGE, CLAIM_PAID`

Целевая переменная `HAS_CLAIM` создаётся автоматически из `CLAIM_PAID`:
- `1` — была страховая выплата
- `0` — выплаты не было

---

## Локальный запуск

```bash
pip install -r requirements.txt

# Обработать следующий батч
python run.py -mode update

# Применить модель к новым данным
python run.py -mode inference -file ./data/source/motor_data11-14lats.csv

# Отчёт о качестве данных и метриках моделей
python run.py -mode summary

# Начать поток заново
python run.py -mode reset
```

---

## Развёртывание через GitHub Actions (CI/CD)

Workflow запускается автоматически при **push** или **pull request** в ветку `main`.

Что происходит при запуске:
1. Устанавливается Python и зависимости из `requirements.txt`
2. Запускается один шаг обучения: `python run.py -mode update`
3. Логи обучения сохраняются как артефакт GitHub Actions (`logs/`)

**Файл workflow:** `.github/workflows/train.yml`

```yaml
name: Train Model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run training batch
        run: python run.py -mode update

      - name: Upload logs
        uses: actions/upload-artifact@v4
        with:
          name: training-logs
          path: |
            logs/
            reports/
            models/metrics_history.json
```

Артефакты (логи, отчёты, история метрик) доступны для скачивания в разделе **Actions** репозитория GitHub после каждого запуска.
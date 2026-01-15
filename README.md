# ASI-ML: Predykcja czasów okrążeń FIA WEC

## Opis projektu

Projekt machine learningowy przewidujący czasy okrążeń w wyścigach FIA World Endurance Championship (WEC) na podstawie historycznych danych wyścigowych z lat 2012-2022.

## Zbiór danych

**Źródło:** https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022.

**Data pobrania:** 2025-10-11.

## Licencja

Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO (https://creativecommons.org/licenses/by-nc-sa/3.0/igo/).

## Inicjalizacja projektu

### 1. Sklonuj repozytorium
```bash
git clone https://github.com/PJ-s28184/asi-project.git
cd asi-project
```

### 2. Utwórz środowisko
```bash
conda env create -f environment.yml
conda activate asi-ml
```

### 3. Uruchom pipeline
```bash
kedro run
```

Albo tylko pipeline data_science:
```bash
kedro run --pipeline data_science
```

## Struktura projektu

```
asi-project/
├── conf/                    # Konfiguracja Kedro
├── data/                    # Dane (nie w git)
│   ├── 01_raw/             # Surowy CSV
│   ├── 02_interim/         # Oczyszczone dane
│   ├── 03_primary/         # Train/test split
│   ├── 06_models/          # Wytrenowane modele
│   └── 08_reporting/       # Metryki
├── notebooks/              # Notebooki Jupyter
│   ├── 01_eda.ipynb        # EDA i czyszczenie
│   └── 02_baseline_ml.ipynb # Baseline model
├── src/asi_project/        # Kod pipeline'u
│   └── pipelines/data_science/
│       ├── nodes.py        # Funkcje (clean, split, train, evaluate)
│       └── pipeline.py     # Definicja pipeline'u
└── tests/                  # Testy jednostkowe
```

## Pipeline Kedro

Pipeline ma 4 kroki:
1. **clean** - czyści dane (usuwa duplikaty, tworzy lap_time_seconds, pit_time_seconds, usuwa braki)
2. **split** - dzieli na train/test (80/20)
3. **train_baseline** - trenuje RandomForest i zapisuje model
4. **evaluate** - oblicza metryki (MAE, RMSE, R², MAPE) i zapisuje je do CSV

Wszystko jest w `src/asi_project/pipelines/data_science/`. Model zapisuje się do `data/06_models/model_baseline.pkl`, metryki do `data/08_reporting/metrics_baseline.csv`.

## Metryki

Model używa MAE jako głównej metryki (bo rozkład czasów okrążeń jest bardzo skośny), plus RMSE, R² i MAPE dla porównania.

## W&B
### Porównanie eksperymentów AutoGluon
| # | Run ID     | Nazwa runa         | MAE      | MAPE     | RMSE      | MSE        | R²        | Runtime | Production Score |
|---|------------|--------------------|----------|----------|-----------|------------|-----------|---------|-------------------|
| 1 | zy1lkdaj   | ethereal-wave-5    | 0.049028 | 0.001546 | 5.084088  | 25.847949  | 0.998689  | 0       | **0.049028**      |
| 2 | dv6yfpyk   | dreadful-rift-1    | 0.049028 | 0.001546 | 5.084088  | 25.847949  | 0.998689  | 0       | **0.049028**      |
| 3 | b4sqcr6w   | rosy-blaze-4       | 0.049144 | 0.001986 | 4.750120  | 22.563638  | 0.998856  | 0       | 0.049144          |
| 4 | l286uhy6   | scarlet-bee-9      | 0.220666 | 0.075311 | 12.391677 | 153.553654 | 0.992407  | 0       | 0.220666          |
| 5 | q3x05q68   | magic-planet-12    | 0.370108 | 0.138567 | 11.821234 | 139.741563 | 0.993090  | 0       | 0.370108          |
| 6 | 4r4v28kj   | prime-darkness-13  | 0.501203 | 0.189422 | 14.602788 | 213.241408 | 0.989456  | 0       | 0.501203          |

<img width="871" height="560" alt="Image" src="https://github.com/user-attachments/assets/01f1c35d-3acd-45e6-aca1-aa8d9e11a13b" />

**W&B:** https://wandb.ai/s27669-polsko-japo-ska-akademia-technik-komputerowych/asi-project.

## Uruchomienie serwisu FastAPI

```bash
uvicorn src.api.main:app --reload --port 8000
```
## test health
```
curl http://127.0.0.1:8000/healthz
```

## predykcja
```
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "number":22,
  "driver_number":7,
  "lap_number":12,
  "kph":150,
  "top_speed":300,
  "season":2022,
  "round":4
}'
```
## podgląd zapisanych predykcji w bazie
```
sqlite3 local.db "SELECT * FROM predictions LIMIT 5;"
```

## Docker quickstart

```bash
docker compose up --build
```

API:
```bash
curl http://localhost:8000/healthz
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"number":22,"driver_number":7,"lap_number":12,"kph":150.0,"top_speed":300.0,"season":2022,"round":4}'
```

UI:
```
open http://localhost:8501
```

DB:
```bash
docker exec -it <container_db> psql -U app -d appdb -c "SELECT * FROM predictions LIMIT 5;"
```


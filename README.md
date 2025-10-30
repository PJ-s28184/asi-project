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

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
### Comparison of AutoGluon experiments
| run_id       | name                  | MAE     | MAPE   | RMSE    | MSE       | R²       | Runtime(s) |
|--------------|---------------------|---------|--------|---------|-----------|----------|------------|
| smi3qwbd     | splendid-lake-1      | 0.0490  | 5.084  | 25.848  | 0.001546  | 0.9987   | 0          |
| zy1lkdaj     | serene-valley-8      | 0.0490  | 5.084  | 25.848  | 0.001546  | 0.9987   | 0          |
| dv6yfpyk     | divine-durian-2      | 0.0490  | 5.084  | 25.848  | 0.001546  | 0.9987   | 0          |
| b4sqcr6w     | ruby-pond-7          | 0.0491  | 4.750  | 22.564  | 0.001986  | 0.9989   | 0          |
| dr324mmx     | happy-sea-5          | 0.0501  | 5.174  | 26.772  | 0.001932  | 0.9986   | 651        |
| nxjupbfk     | major-plasma-3       | 0.0554  | 6.451  | 41.618  | 0.001615  | 0.9979   | 0          |
| gkb8bskl     | sandy-sun-4          | 0.0554  | 6.451  | 41.618  | 0.001615  | 0.9979   | 0          |
| l286uhy6     | vibrant-armadillo-12 | 0.2207  | 12.392 | 153.554 | 0.075311  | 0.9924   | 0          |

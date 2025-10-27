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

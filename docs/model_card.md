# Model Card
## Problem & Intended Use
Predict lap time (seconds) for WEC laps. Intended for exploratory modeling and feature analysis.

## Data
- Source: https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022
- Size: 504000 rows 
- PII: none
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO (https://creativecommons.org/licenses/by-nc-sa/3.0/igo/)
## Metrics (main + secondary, test split)
- Main metric: MSE (test split)
- Baseline : MAE = 0.055392674
- AutoGluon best: MAE = 0.049028
## Limitations & Risks
- Data covers certain tracks and conditions — may not generalize to others.
- Model may be biased if features correlate with unobserved confounders.
- Predictions may be invalid for extreme unseen configurations.
## Versioning
- W&B run:https://wandb.ai/<team-name>/asi-project/runs/smi3qwbd
- Model artifact: <team-name>/run-smi3qwbd-history:production (v3) <link>
- Code: commit 4f7a2c9 <link>
- Data: clean_data:v12 <link>
- Env: Python 3.11, AutoGluon 1.4.0, sklearn 1.5
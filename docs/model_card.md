# Model Card
## Problem & Intended Use
Predict lap time (seconds) for WEC laps. Intended for exploratory modeling and feature analysis.

## Data
- Source: https://www.kaggle.com/datasets/tristenterracciano/fia-wec-lap-data-20122022
- Size: 379375 rows
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
- W&B run:https://wandb.ai/s27669-polsko-japo-ska-akademia-technik-komputerowych/asi-project/veb48587
- Model artifact: s27669-polsko-japo-ska-akademia-technik-komputerowych/ag_model:production (v3) 
- Code: commit 4f7a2c9 <link>
- Data: clean_data:v12 <link>
- Env: Python 3.11, AutoGluon 1.4.0, sklearn 1.5

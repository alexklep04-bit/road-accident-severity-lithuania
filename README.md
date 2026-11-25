# Road accident severity in Lithuania

Individual project for the course Deep Learning Systems (INF4039).

The goal is to predict if a road traffic accident with injured people in Lithuania ends with someone killed or seriously injured. I use the police data from data.gov.lt (2021-2025) and compare a random forest, a multilayer perceptron (MLP) and a Transformer for tabular data (FT-Transformer).

The main notebook is `road_accident_severity_LT.ipynb` (all outputs are saved, so it can be read without running).

## Data

- Source: Lithuanian Police Department, [data.gov.lt dataset 509](https://data.gov.lt/datasets/509/), license CC BY 4.0
- The notebook downloads the raw JSON files by itself (about 456 MB, not included in this repository)
- `data/processed/accidents_clean.csv` is the cleaned dataset: 14 049 injury accidents, 59 features and the target `severe`

## Results (test year 2025)

| Model | ROC-AUC |
|---|---|
| Logistic regression (baseline) | 0.727 |
| Random forest | 0.728 |
| MLP | 0.735 |
| FT-Transformer | 0.735 |
| Ensemble of the three models | 0.739 |

H1 (ROC-AUC >= 0.70) is accepted. H2 (neural networks are better than the random forest) is not confirmed statistically: the networks are slightly better, but the difference is inside the bootstrap confidence interval. The most important feature for all models is the age of the oldest pedestrian / cyclist involved in the accident.

## How to run

- Google Colab: open the notebook and use Runtime -> Run all (around 1-1.5 hours on CPU, faster with GPU)
- Locally: Python 3.12, `pip install -r requirements.txt`, then run the notebook

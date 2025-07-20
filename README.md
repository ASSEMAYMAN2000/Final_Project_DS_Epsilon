# Medical Appointment No-Show Prediction

This project predicts whether patients will miss medical appointments ("no-show") using the Kaggle Medical Appointment No Shows dataset. The goal is to help clinics optimize scheduling with a machine learning model and a Streamlit app.

## Overview
- **Objective**: Predict no-shows with precision and recall ≥ 0.3 (epsilon threshold).
- **Dataset**: `KaggleV2-May-2016.csv` (~71,959 rows after cleaning).
- **Steps**:
  - Cleaned data (fixed typos, removed invalid ages/dates, encoded variables).
  - Performed EDA with 6 visualizations (histograms, bar plots, heatmap).
  - Created `WaitingDays` feature and selected 6 features (Age, WaitingDays, etc.).
  - Trained and tuned Random Forest (precision: 0.35, recall: 0.56).
  - Validated with 5-fold cross-validation.
  - Deployed a Streamlit app for predictions.
- **Results**: Achieved epsilon threshold (0.3); app predicts no-show risk.

## Repository Contents
- `Notebook.ipynb`: Jupyter Notebook with all steps.
- `app.py`: Streamlit app for no-show predictions.
- `requirements.txt`: App dependencies.
- `model.pkl`: Tuned Random Forest model.
- `*.png`: EDA visualizations (age histogram, etc.).



# Automated ML Toolkit

A complete end-to-end machine learning toolkit that automates EDA, data cleaning, and model training.

## One Command Does Everything

## What it does
1. **EDA** — dataset overview, missing values, statistical summary, distribution plots, correlation matrix, PDF report
2. **Data Cleaning** — removes duplicates, handles missing values, removes outliers, encodes categorical columns
3. **Model Training** — auto-detects regression vs classification, trains XGBoost, saves model

## Files
- `eda.py` — run EDA and generate PDF report
- `cleaner.py` — clean any CSV dataset
- `pipeline.py` — full pipeline in one command

## Tech Stack
Python • Pandas • NumPy • Matplotlib • Seaborn • XGBoost • Scikit-learn • fpdf2

## Example Output
- Full EDA PDF report
- Cleaned CSV file
- Trained XGBoost model saved as .joblib
- RMSE or Accuracy score

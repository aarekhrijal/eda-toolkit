# Automated EDA Toolkit

A Python tool that automatically generates a full Exploratory Data Analysis report for any CSV dataset.

## What it does
- Basic dataset information (rows, columns, data types)
- Missing values analysis with percentages
- Statistical summary of all numeric columns
- Distribution plots (histogram + boxplot) for every numeric column
- Correlation matrix heatmap

## How to use
1. Install requirements: `pip install pandas numpy matplotlib seaborn`
2. Update the filepath in `eda.py` to point to your CSV
3. Run: `python eda.py`
4. Charts are saved automatically to `eda_output/` folder

## Example Output
- Missing values report
- Statistical summary
- Distribution plots for each numeric column
- Correlation matrix showing feature relationships

## Tech Stack
Python • Pandas • NumPy • Matplotlib • Seaborn
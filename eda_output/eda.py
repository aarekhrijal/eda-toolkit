import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF

def lead_data(filepath):
    df = pd.read_csv(filepath)
    return df


def basic_info(df):
    print("="*50)
    print("BASIC INFORMATION")
    print("="*50)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")

def missing_values(df):
    print("="*50)
    print("MISSING VALUES")
    print("="*50)
    missing = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) *100
    missing_df = pd.DataFrame({
        'Missing Count' : missing,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0 ]
    if len(missing_df) ==0:
        print("No missing values found!")
    else:
        print(missing_df)


def statistical_summary(df) :
    print("="*50)
    print("STATISTICAL SUMMARY")
    print("="* 50)
    print(df.describe())

def plot_distributions(df, output_folder="eda_output"):
    os.makedirs(output_folder, exist_ok = True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(y= df[col])
        plt.title(f'Boxplot of {col}')

        plt.tight_layout()
        plt.savefig(f"{output_folder}/{col}_distribution.png")
        plt.close()

    print(f"Distribution plots saved to {output_folder}/")


def correlation_matrix(df, output_folfer="eda_output"):
    os.makedirs(output_folfer, exist_ok=True)
    numeric_cols = df.select_dtypes(include = [np.number])

    if len(numeric_cols.columns) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_folfer}/correlation_matrix.png")
        plt.close()
        print("Correlation matrix saved.")
    else:
        print("Not enough numeric columns for correlation matrix.")

def generate_pdf_report(df, output_folder="eda_output"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "EDA Report", ln=True, align="C")
    pdf.ln(5)
    
    # Dataset Overview
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Dataset Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Rows: {df.shape[0]}", ln=True)
    pdf.cell(0, 8, f"Columns: {df.shape[1]}", ln=True)
    pdf.ln(5)
    
    # Missing Values
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "2. Missing Values", ln=True)
    pdf.set_font("Helvetica", "", 10)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        pdf.cell(0, 8, "No missing values found.", ln=True)
    else:
        for col, count in missing.items():
            percent = round((count / len(df)) * 100, 2)
            pdf.cell(0, 8, f"{col}: {count} missing ({percent}%)", ln=True)
    pdf.ln(5)
    
    # Distribution Charts
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "3. Distribution Plots", ln=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        img_path = f"{output_folder}/{col}_distribution.png"
        if os.path.exists(img_path):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 8, col, ln=True)
            pdf.image(img_path, w=180)
            pdf.ln(3)
    
    # Correlation Matrix
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "4. Correlation Matrix", ln=True)
    corr_path = f"{output_folder}/correlation_matrix.png"
    if os.path.exists(corr_path):
        pdf.image(corr_path, w=180)
    
    pdf.output(f"{output_folder}/eda_report.pdf")
    print(f"PDF report saved to {output_folder}/eda_report.pdf")
    
def run_eda(filepath):
    print(f"\Running EDA on: {filepath}\n")
    df = lead_data(filepath)
    basic_info(df)
    missing_values(df)
    statistical_summary(df)
    plot_distributions(df)
    correlation_matrix(df)
    print("\nDEA Complete!")
    generate_pdf_report(df)

if __name__=="__main__":
    run_eda(r"C:\AI,ML\train.csv")


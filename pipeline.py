import pandas as pd
import numpy as np
import argparse
from eda import run_eda
from cleaner import clean_data
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


def detect_problem_type(df, target_col):
    unique_values = df[target_col].nunique()
    if unique_values <= 10:
        return "classification"
    else:
        return "regression"
    

def train_model(df, target_col, problem_type):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
    )

    if problem_type == "regression":
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE: {score:.4f}")

    else:
        model = XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        print(f"Accuracy: {score:.4f}")

    return model, X_test, y_test     

def evaluate_model(model, X_test, y_test, problem_type, output_folder="pipeline_output"):
    os.makedirs(output_folder, exist_ok = True)
    predictions = model.predict(X_test)

    if problem_type == "regression":
    
        # Actual vs Predicted plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/actual_vs_predicted.png")
        plt.close()

        print("Evaluation charts saved.")

    else:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/confusion_matrix.png")
        plt.close()
        print(classification_report(y_test, predictions))
        print("Confusion matrix saved.")


def run_pipeline(filepath, target_col):
    print("="*50)
    print("ML PIPELINE STARTING")
    print("="*50)

    # Step 1: EDA
    print("\nStep 1: Running EDA...")
    run_eda(filepath)
    
    # Step 2: Clean data
    print("\nStep 2: Cleaning data...")
    df = clean_data(filepath, target_col=target_col)
    
    # Step 3: Detect problem type
    problem_type = detect_problem_type(df, target_col)
    print(f"\nProblem type detected: {problem_type}")
    
    # Step 4: Train model
    print("\nStep 3: Training model...")
    model, X_test, y_test = train_model(df, target_col, problem_type)

    # Step 5: Evaluate
    print("\nStep 4: Evaluating model...")
    evaluate_model(model, X_test, y_test, problem_type)
        
    # Step 6: Save model
    os.makedirs("pipeline_output", exist_ok=True)
    joblib.dump(model, "pipeline_output/model.joblib")
    print("\nModel saved to pipeline_output/model.joblib")
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    args = parser.parse_args()
    
    run_pipeline(args.data, args.target)
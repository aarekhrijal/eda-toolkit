import pandas as pd
import numpy as np

def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Duplicates removed: {before - after}")
    return df

def handle_missing_values(df, strategy="mean"):
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy ==  "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    print(f"Missing values handled using strategy: {strategy}")
    return df


def remove_outliers(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    before = len(df)
    
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]
    
    after = len(df)
    print(f"Outliers removed: {before - after} rows")
    return df


def encode_categorical(df):
    categorical_cols = df.select_dtypes(include = ["object"]).columns

    for col in categorical_cols:
        if df[col]. nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies],axis = 1)
            df = df.drop(col, axis = 1)
        else:
            df[col] = pd.factorize(df[col])[0]

    print(f"Categorical columns encoded: {len(categorical_cols)}")
    return df


def clean_data(filepath, target_col=None, strategy="mean", remove_out=True):
    print(f"\nCleaning: {filepath}\n")
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    df = drop_duplicates(df)
    df = handle_missing_values(df, strategy=strategy)
    
    if remove_out:
        df = remove_outliers(df)
    
    df = encode_categorical(df)
    
    print(f"\nFinal shape: {df.shape}")
    
    output_path = filepath.replace(".csv", "_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    clean_data(r"C:\AI,ML\train.csv", target_col="SalePrice")

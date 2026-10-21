"""
Module: preprocessing.py
Description: Step 3 of Data Science - Data Cleaning, Sanity Checks, Outlier Analysis, and Preprocessing.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "flood_cleaned.csv")


def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing or null values in each column.
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percent
    })
    return missing_df


def inspect_duplicates(df: pd.DataFrame) -> int:
    """
    Identifies duplicate records in the dataset.
    """
    num_duplicates = df.duplicated().sum()
    return int(num_duplicates)


def detect_outliers_iqr(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Detects outliers using the standard Interquartile Range (IQR) method:
    Lower Bound = Q1 - 1.5 * IQR, Upper Bound = Q3 + 1.5 * IQR.
    """
    outlier_stats = []
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_stats.append({
            "Feature": col,
            "Q1": round(float(q1), 3),
            "Q3": round(float(q3), 3),
            "IQR": round(float(iqr), 3),
            "Lower_Bound": round(float(lower_bound), 3),
            "Upper_Bound": round(float(upper_bound), 3),
            "Outlier_Count": int(len(outliers)),
            "Outlier_Percentage": round(float(len(outliers) / len(df) * 100), 2)
        })
    return pd.DataFrame(outlier_stats)


def clean_and_preprocess(df: pd.DataFrame, save_path: str = PROCESSED_DATA_PATH) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end cleaning pipeline:
    1. Duplicate removal (if any).
    2. Missing value imputation / validation.
    3. Type verification.
    4. Outlier profiling.
    5. Saves cleaned dataset to data/processed.
    """
    logger.info("Initiating Step 3: Data Cleaning and Preprocessing...")
    initial_shape = df.shape
    
    # 1. Check & handle duplicates
    dup_count = inspect_duplicates(df)
    logger.info(f"Duplicate rows detected: {dup_count}")
    cleaned_df = df.drop_duplicates().copy()
    
    # 2. Check & handle missing values
    missing_report = inspect_missing_values(cleaned_df)
    total_missing = missing_report["Missing_Count"].sum()
    logger.info(f"Total missing values found: {total_missing}")
    if total_missing > 0:
        logger.info("Imputing missing values using median strategy...")
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    
    # 3. Ensure numeric types
    for col in cleaned_df.columns:
        if col != "FloodProbability":
            cleaned_df[col] = cleaned_df[col].astype(int)
        else:
            cleaned_df[col] = cleaned_df[col].astype(float)
            
    # 4. Outlier detection
    feature_cols = [c for c in cleaned_df.columns if c != "FloodProbability"]
    outlier_df = detect_outliers_iqr(cleaned_df, feature_cols)
    total_outliers = outlier_df["Outlier_Count"].sum()
    logger.info(f"Outlier analysis completed: {total_outliers} total anomaly flags across features.")
    logger.info("Environmental rationale: Extreme meteorological/topographical readings are valid high-risk events and retained for training.")

    # 5. Save cleaned dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cleaned_df.to_csv(save_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to: {save_path}")

    summary = {
        "Initial_Shape": initial_shape,
        "Final_Shape": cleaned_df.shape,
        "Duplicates_Removed": dup_count,
        "Missing_Values_Found": int(total_missing),
        "Outlier_Summary": outlier_df.to_dict(orient="records")
    }
    return cleaned_df, summary


if __name__ == "__main__":
    raw_df = load_dataset()
    cleaned_df, report = clean_and_preprocess(raw_df)
    print("Preprocessing Report Summary:")
    print(f"Shape: {report['Final_Shape']}, Duplicates: {report['Duplicates_Removed']}, Missing: {report['Missing_Values_Found']}")

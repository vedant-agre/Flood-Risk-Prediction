"""
Module: features.py
Description: Step 5 of Data Science - Feature Engineering, Domain Index Computation, Scaling, and Selection.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset
from src.preprocessing import clean_and_preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

BASE_FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
]

RISK_BINS = [0.0, 0.470, 0.500, 0.530, 1.0]
RISK_LABELS = ["Low", "Moderate", "High", "Severe"]


def create_risk_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretizes the continuous FloodProbability into 4 balanced multi-class risk categories:
    Low (<0.470), Moderate (0.470-0.500), High (0.500-0.530), Severe (>0.530).
    """
    df_out = df.copy()
    df_out["Risk_Level"] = pd.cut(df_out["FloodProbability"], bins=RISK_BINS, labels=RISK_LABELS)
    return df_out


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-specific composite indices and statistical aggregations.
    """
    df_feat = df.copy()
    
    # 1. Hydrological Stress Ratio (Precipitation vs Drainage Conveyance)
    df_feat["Hydrological_Stress_Ratio"] = (
        (df_feat["MonsoonIntensity"] + 1.0) / 
        (df_feat["TopographyDrainage"] + df_feat["DrainageSystems"] + 1.0)
    )
    
    # 2. Environmental Degradation Index (Deforestation, Urbanization, Wetland Loss, Encroachments)
    df_feat["Environmental_Degradation_Index"] = (
        df_feat["Deforestation"] + df_feat["Urbanization"] + 
        df_feat["WetlandLoss"] + df_feat["Encroachments"]
    ) / 4.0
    
    # 3. Infrastructure Resilience Index (Defenses vs Degradation)
    df_feat["Infrastructure_Resilience_Index"] = (
        (df_feat["DamsQuality"] + df_feat["RiverManagement"] + df_feat["DrainageSystems"]) / 3.0 -
        0.5 * df_feat["DeterioratingInfrastructure"]
    )
    
    # 4. Human Vulnerability & Exposure Index
    df_feat["Human_Exposure_Index"] = (
        df_feat["PopulationScore"] * (1.0 + (df_feat["IneffectiveDisasterPreparedness"] / 10.0))
    )
    
    # Note on Feature Leakage Audit:
    # A mathematical audit of Kaggle s4e5 reveals that naive row-wise sum/mean directly leaks 
    # the underlying linear synthetic generator. In rigorous academic data science practice,
    # trivial row sums are excluded to evaluate genuine multi-factor environmental interaction learning.
    
    return df_feat


def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Computes Mutual Information scores between features and the categorical target.
    """
    logger.info("Computing Mutual Information feature importance...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "Mutual_Information": mi_scores
    }).sort_values(by="Mutual_Information", ascending=False).reset_index(drop=True)
    return mi_df


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler, LabelEncoder
]:
    """
    Executes the full Feature Engineering pipeline:
    1. Creates multi-class risk categories.
    2. Builds domain & statistical features.
    3. Encodes labels.
    4. Performs stratified train/test split.
    5. Scales features using StandardScaler and serializes objects for inference.
    """
    logger.info("Initiating Step 5: Feature Engineering and Dataset Preparation...")
    
    # 1. Target Categorization
    df_categorized = create_risk_classes(df)
    
    # 2. Feature Engineering
    df_featured = engineer_domain_features(df_categorized)
    
    # Exclude targets from input features
    exclude_cols = ["FloodProbability", "Risk_Level"]
    feature_cols = [c for c in df_featured.columns if c not in exclude_cols]
    
    X = df_featured[feature_cols]
    y_raw = df_featured["Risk_Level"]
    
    # 3. Label Encoding (Low: 0, Moderate: 1, High: 2, Severe: 3)
    le = LabelEncoder()
    le.fit(RISK_LABELS)  # Ensures explicit order
    y = le.transform(y_raw)
    
    # 4. Stratified Split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    # Save artifacts for deployment & reproducibility
    joblib.dump(scaler, os.path.join(MODELS_DIR, "flood_scaler.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_names.joblib"))
    
    # Save processed featured CSV
    df_featured.to_csv(os.path.join(PROCESSED_DIR, "flood_featured.csv"), index=False)
    
    # Save train & test splits
    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df["Risk_Level"] = y_train
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df["Risk_Level"] = y_test
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    logger.info(f"Feature engineering completed: {X.shape[1]} total features created.")
    logger.info(f"Train samples: {X_train_scaled.shape[0]}, Test samples: {X_test_scaled.shape[0]}")
    logger.info("Scaler and Encoder serialized to models/.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, le


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/flood.csv")
    cleaned_df, _ = clean_and_preprocess(raw_df)
    X_train, X_test, y_train, y_test, features, scaler, le = prepare_datasets(cleaned_df)
    print("Feature List:")
    for idx, f in enumerate(features):
        print(f"  {idx+1}. {f}")

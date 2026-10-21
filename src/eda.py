"""
Module: eda.py
Description: Step 4 of Data Science - Comprehensive Exploratory Data Analysis (EDA) and Visualization.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def calculate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical indicators including mean, std, quantiles, skewness, and kurtosis.
    """
    stats_df = df.describe().T
    stats_df["skewness"] = df.skew()
    stats_df["kurtosis"] = df.kurtosis()
    return stats_df


def plot_target_distribution(df: pd.DataFrame, save_path: str = os.path.join(FIGURES_DIR, "01_target_distribution.png")):
    """
    Visualizes continuous target distribution and discretized flood risk classes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Continuous distribution
    sns.histplot(df["FloodProbability"], kde=True, ax=axes[0], color="#2b5c8f", bins=30, edgecolor="black", alpha=0.7)
    mean_val = df["FloodProbability"].mean()
    median_val = df["FloodProbability"].median()
    axes[0].axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")
    axes[0].axvline(median_val, color="green", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.3f}")
    axes[0].set_title("Distribution of Flood Probability", fontweight="bold")
    axes[0].set_xlabel("Flood Occurrence Probability (0.0 to 1.0)")
    axes[0].set_ylabel("Frequency Count")
    axes[0].legend()
    
    # Categorical Risk Tiers
    bins = [0.0, 0.470, 0.500, 0.530, 1.0]
    labels = ["Low Risk", "Moderate Risk", "High Risk", "Severe Risk"]
    risk_tiers = pd.cut(df["FloodProbability"], bins=bins, labels=labels)
    tier_counts = risk_tiers.value_counts()[labels]
    
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    bars = axes[1].bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 150, f"{yval:,}\n({yval/len(df)*100:.1f}%)", 
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_title("Categorical Flood Risk Level Breakdown", fontweight="bold")
    axes[1].set_xlabel("Risk Classification Tier")
    axes[1].set_ylabel("Sample Count")
    axes[1].set_ylim(0, max(tier_counts.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Target distribution plot saved to: {save_path}")


def plot_feature_distributions(df: pd.DataFrame, save_path: str = os.path.join(FIGURES_DIR, "02_feature_distributions.png")):
    """
    Subplots showing histograms for all 20 environmental and rainfall features.
    """
    feature_cols = [c for c in df.columns if c != "FloodProbability"]
    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], discrete=True, ax=axes[i], color="#34495e", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Indicator Score")
        axes[i].set_ylabel("Count")
        
    plt.suptitle("Univariate Distributions of 20 Rainfall & Environmental Indicators", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Feature distributions plot saved to: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = os.path.join(FIGURES_DIR, "03_correlation_heatmap.png")):
    """
    Generates Pearson correlation heatmap across all variables.
    """
    corr = df.corr()
    plt.figure(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.2, vmax=0.4, annot=True, fmt=".2f",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Pearson Correlation Matrix (All Indicators & Target)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Correlation heatmap saved to: {save_path}")


def plot_target_correlations(df: pd.DataFrame, save_path: str = os.path.join(FIGURES_DIR, "04_target_correlations.png")):
    """
    Bar chart of correlations specifically with FloodProbability.
    """
    corr = df.corr()["FloodProbability"].drop("FloodProbability").sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    colors = ["#2980b9" if val > 0 else "#c0392b" for val in corr.values]
    bars = plt.barh(corr.index, corr.values, color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.003, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                 va="center", fontsize=9, fontweight="bold")
    plt.title("Feature Correlation with Flood Probability", fontsize=12, fontweight="bold")
    plt.xlabel("Pearson Correlation Coefficient (r)")
    plt.xlim(0, max(corr.values) + 0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Target correlations plot saved to: {save_path}")


def plot_bivariate_interactions(df: pd.DataFrame, save_path: str = os.path.join(FIGURES_DIR, "05_bivariate_interactions.png")):
    """
    Bivariate interaction boxplots showing how indicators shift across Risk Tiers.
    """
    bins = [0.0, 0.470, 0.500, 0.530, 1.0]
    labels = ["Low", "Moderate", "High", "Severe"]
    df_copy = df.copy()
    df_copy["Risk_Tier"] = pd.cut(df_copy["FloodProbability"], bins=bins, labels=labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    
    sns.boxplot(data=df_copy, x="Risk_Tier", y="MonsoonIntensity", hue="Risk_Tier", legend=False, ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Monsoon Intensity by Flood Risk Tier", fontweight="bold")
    axes[0, 0].set_ylabel("Monsoon Intensity Score")
    
    sns.boxplot(data=df_copy, x="Risk_Tier", y="TopographyDrainage", hue="Risk_Tier", legend=False, ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title("Topography Drainage by Flood Risk Tier", fontweight="bold")
    axes[0, 1].set_ylabel("Topography Drainage Score")
    
    sns.boxplot(data=df_copy, x="Risk_Tier", y="Deforestation", hue="Risk_Tier", legend=False, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Deforestation by Flood Risk Tier", fontweight="bold")
    axes[1, 0].set_ylabel("Deforestation Rate")
    
    sns.boxplot(data=df_copy, x="Risk_Tier", y="DeterioratingInfrastructure", hue="Risk_Tier", legend=False, ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title("Deteriorating Infrastructure by Flood Risk Tier", fontweight="bold")
    axes[1, 1].set_ylabel("Infrastructure Decay Score")
    
    plt.suptitle("Bivariate Analysis: Key Environmental Indicators across Risk Levels", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Bivariate interactions plot saved to: {save_path}")


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes full Step 4 Exploratory Data Analysis workflow.
    """
    logger.info("Executing Step 4: Exploratory Data Analysis (EDA)...")
    stats_df = calculate_descriptive_stats(df)
    
    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_correlations(df)
    plot_bivariate_interactions(df)
    
    logger.info("Step 4 EDA completed successfully. Visual figures generated in reports/figures/.")
    return stats_df


if __name__ == "__main__":
    from src.preprocessing import clean_and_preprocess
    raw_df = pd.read_csv("data/raw/flood.csv")
    cleaned_df, _ = clean_and_preprocess(raw_df)
    stats = run_eda(cleaned_df)
    print("Descriptive Statistics Preview:")
    print(stats.iloc[:5, :6])

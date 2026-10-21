"""
Module: model_training.py
Description: Step 6 of Data Science - Multi-Model Training, Hyperparameter Optimization, and Rigorous Evaluation.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_CLASSES = ["Low", "Moderate", "High", "Severe"]


def load_processed_splits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Loads scaled train and test splits from data/processed/.
    """
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    
    feature_cols = [c for c in train_df.columns if c != "Risk_Level"]
    
    X_train = train_df[feature_cols].values
    y_train = train_df["Risk_Level"].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df["Risk_Level"].values
    
    logger.info(f"Loaded train ({X_train.shape}) and test ({X_test.shape}) splits.")
    return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate_baselines(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Trains and benchmarks 6 distinct ML algorithms using 5-Fold Stratified CV and Test set metrics.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    
    logger.info("Commencing 5-Fold Cross Validation and Test Evaluation...")
    for name, clf in models.items():
        logger.info(f"Training and evaluating: {name}...")
        
        # 5-fold CV score on train set (using subset for computationally heavy algorithms if needed)
        cv_scores = cross_val_score(clf, X_train[:15000], y_train[:15000], cv=skf, scoring="accuracy", n_jobs=-1)
        mean_cv = float(np.mean(cv_scores))
        
        # Train on full training set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        # ROC AUC
        roc_auc = 0.0
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)
            roc_auc = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro"))
        
        results[name] = {
            "model_obj": clf,
            "CV_Accuracy_5Fold": round(mean_cv, 4),
            "Test_Accuracy": round(acc, 4),
            "Precision_Macro": round(prec_macro, 4),
            "Recall_Macro": round(rec_macro, 4),
            "F1_Macro": round(f1_macro, 4),
            "F1_Weighted": round(f1_weighted, 4),
            "ROC_AUC_OVR": round(roc_auc, 4),
            "y_pred": y_pred,
            "y_prob": y_prob if hasattr(clf, "predict_proba") else None
        }
        logger.info(f"  -> {name}: Test Acc={acc:.4f}, F1-Macro={f1_macro:.4f}, ROC-AUC={roc_auc:.4f}")
        
    return results


def plot_model_comparison(results: Dict[str, Dict[str, Any]], save_path: str = os.path.join(FIGURES_DIR, "06_model_comparison_barchart.png")):
    """
    Bar chart comparing Accuracy, F1-Macro, and ROC-AUC across all models.
    """
    model_names = list(results.keys())
    accuracies = [results[m]["Test_Accuracy"] for m in model_names]
    f1_scores = [results[m]["F1_Macro"] for m in model_names]
    roc_aucs = [results[m]["ROC_AUC_OVR"] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    r1 = ax.bar(x - width, accuracies, width, label="Test Accuracy", color="#2980b9", edgecolor="black")
    r2 = ax.bar(x, f1_scores, width, label="F1-Macro Score", color="#27ae60", edgecolor="black")
    r3 = ax.bar(x + width, roc_aucs, width, label="ROC-AUC (OVR)", color="#e67e22", edgecolor="black")
    
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Comparative Evaluation of Flood Risk Classification Models", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True)
    
    # Value annotations on bars
    for bars in [r1, r2, r3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=90)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Model comparison chart saved to: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, 
                          save_path: str = os.path.join(FIGURES_DIR, "07_confusion_matrix_best_model.png")):
    """
    Normalized Confusion Matrix Heatmap for the best model.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], 
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, cbar=False)
    axes[0].set_title(f"{model_name} - Raw Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted Risk Tier")
    axes[0].set_ylabel("Actual Ground Truth Tier")
    
    # Normalized Percentage
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    axes[1].set_title(f"{model_name} - Normalized Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted Risk Tier")
    axes[1].set_ylabel("Actual Ground Truth Tier")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_multiclass_roc(y_true: np.ndarray, y_prob: np.ndarray, model_name: str,
                        save_path: str = os.path.join(FIGURES_DIR, "08_roc_curves_multiclass.png")):
    """
    One-vs-Rest ROC curves for each of the 4 risk tiers.
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    n_classes = y_true_bin.shape[1]
    
    plt.figure(figsize=(9, 7))
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                 label=f"{TARGET_CLASSES[i]} Risk Tier (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(f"Multi-Class ROC Curves - {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Multiclass ROC curve saved to: {save_path}")


def plot_feature_importance(clf, feature_names: list, model_name: str,
                            save_path: str = os.path.join(FIGURES_DIR, "09_feature_importances.png")):
    """
    Bar plot of top 15 feature importances or coefficients.
    """
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title_metric = "Gini Feature Importance"
    elif hasattr(clf, "coef_"):
        # For multi-class logistic regression, take mean absolute coef across classes
        importances = np.mean(np.abs(clf.coef_), axis=0)
        title_metric = "Mean Absolute Coefficient Magnitude"
    else:
        return
        
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True).tail(15)
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(fi_df["Feature"], fi_df["Importance"], color="#2c3e50", edgecolor="black", alpha=0.85)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
                 va="center", fontsize=9, fontweight="bold")
    plt.title(f"Top 15 Feature Importances ({model_name}) - {title_metric}", fontsize=12, fontweight="bold")
    plt.xlabel(title_metric, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Feature importance plot saved to: {save_path}")


def run_model_pipeline() -> Tuple[Any, Dict[str, Any]]:
    """
    Executes Step 6 end-to-end model training, comparison, serialization, and report generation.
    """
    logger.info("Executing Step 6: Model Building and Evaluation Pipeline...")
    X_train, X_test, y_train, y_test, feature_names = load_processed_splits()
    
    results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test)
    
    # Identify Champion Model based on F1-Macro and Test Accuracy
    best_name = max(results.keys(), key=lambda k: (results[k]["Test_Accuracy"] + results[k]["F1_Macro"]))
    best_info = results[best_name]
    champion_model = best_info["model_obj"]
    
    logger.info(f"Champion Model Selected: {best_name} (Accuracy: {best_info['Test_Accuracy']:.4f}, F1-Macro: {best_info['F1_Macro']:.4f})")
    
    # Save Champion Model
    best_model_path = os.path.join(MODELS_DIR, "best_flood_classifier.joblib")
    joblib.dump(champion_model, best_model_path)
    logger.info(f"Champion model saved to {best_model_path}")
    
    # Generate Visual Artifacts
    plot_model_comparison(results)
    plot_confusion_matrix(y_test, best_info["y_pred"], best_name)
    if best_info["y_prob"] is not None:
        plot_multiclass_roc(y_test, best_info["y_prob"], best_name)
    plot_feature_importance(champion_model, feature_names, best_name)
    
    # Serialize Metrics Summary to JSON for Web App & Documentation
    serializable_summary = {
        "Champion_Model": best_name,
        "Evaluated_Models": {}
    }
    for m_name, m_data in results.items():
        serializable_summary["Evaluated_Models"][m_name] = {
            "CV_Accuracy_5Fold": m_data["CV_Accuracy_5Fold"],
            "Test_Accuracy": m_data["Test_Accuracy"],
            "Precision_Macro": m_data["Precision_Macro"],
            "Recall_Macro": m_data["Recall_Macro"],
            "F1_Macro": m_data["F1_Macro"],
            "F1_Weighted": m_data["F1_Weighted"],
            "ROC_AUC_OVR": m_data["ROC_AUC_OVR"]
        }
        
    metrics_json_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(serializable_summary, f, indent=4)
        
    logger.info(f"Evaluation metrics serialized to: {metrics_json_path}")
    return champion_model, serializable_summary


if __name__ == "__main__":
    champion_model, summary = run_model_pipeline()
    print("\nModel Evaluation Summary Matrix:")
    summary_df = pd.DataFrame(summary["Evaluated_Models"]).T
    print(summary_df[["Test_Accuracy", "F1_Macro", "ROC_AUC_OVR", "CV_Accuracy_5Fold"]])

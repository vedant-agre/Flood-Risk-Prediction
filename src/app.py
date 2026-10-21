"""
Module: app.py
Description: Step 7 of Data Science - Interactive Web Application & Decision Support Tool.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import BASE_FEATURES, RISK_LABELS, engineer_domain_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Load artifacts
try:
    classifier = joblib.load(os.path.join(MODELS_DIR, "best_flood_classifier.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "flood_scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    with open(os.path.join(MODELS_DIR, "evaluation_metrics.json"), "r") as f:
        evaluation_metrics = json.load(f)
    logger.info("Successfully loaded model, scaler, and metadata.")
except Exception as e:
    logger.error(f"Error loading model artifacts: {e}")
    classifier, scaler, label_encoder, feature_names, evaluation_metrics = None, None, None, None, {}


def generate_mitigation_actions(inputs: dict) -> list:
    """
    Formulates targeted civil and environmental mitigation advisories based on extreme values.
    """
    actions = []
    
    if inputs.get("MonsoonIntensity", 0) >= 8:
        actions.append("🚨 **Extreme Rainfall Alert**: Activate regional retention reservoirs and issue pre-emptive flood stage alerts to riverside communities.")
    if inputs.get("TopographyDrainage", 10) <= 3 or inputs.get("DrainageSystems", 10) <= 3:
        actions.append("🌊 **Conveyance Bottleneck**: Deploy mobile stormwater pumping stations to low-lying drainage choke points.")
    if inputs.get("Deforestation", 0) >= 7 or inputs.get("WetlandLoss", 0) >= 7:
        actions.append("🌱 **Ecological Catchment Action**: Halt vegetation clearance in upper watersheds and initiate emergency wetland buffer restoration.")
    if inputs.get("DamsQuality", 10) <= 3:
        actions.append("🏗️ **Hydraulic Hazard**: Inspect dam spillway mechanisms and mandate controlled reservoir volume discharge.")
    if inputs.get("Siltation", 0) >= 8:
        actions.append("🚜 **Sediment Alert**: Expedite mechanical dredging of silted main channels to restore baseline riverbed capacity.")
    if inputs.get("IneffectiveDisasterPreparedness", 0) >= 7:
        actions.append("📡 **Disaster Readiness**: Mobilize NDRF/SDRF emergency communication channels and designate emergency highland shelters.")
    if inputs.get("Encroachments", 0) >= 7:
        actions.append("⚠️ **Floodplain Encroachment**: Clear temporary blockages on natural floodways and reinforce river levee boundaries.")
        
    if not actions:
        actions.append("✅ **Standard Operational Protocol**: All monitored environmental factors are within baseline safety thresholds. Maintain regular sensor telemetry.")
        
    return actions


@app.route("/")
def home():
    return render_template("index.html", metrics=evaluation_metrics, base_features=BASE_FEATURES)


@app.route("/reports/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}
        input_dict = {}
        for feat in BASE_FEATURES:
            val = float(data.get(feat, 5.0))
            input_dict[feat] = max(0.0, min(20.0, val))
            
        df_input = pd.DataFrame([input_dict])
        
        # Engineer domain features
        df_featured = engineer_domain_features(df_input)
        
        # Align with training features
        X_df = df_featured[feature_names]
        X_scaled = scaler.transform(X_df)
        
        # Predict class and probabilities
        pred_class_idx = int(classifier.predict(X_scaled)[0])
        pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
        
        probs = classifier.predict_proba(X_scaled)[0]
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }
        
        # Calculate continuous probability estimate (weighted expectation across classes)
        # Low: ~0.43, Moderate: ~0.485, High: ~0.515, Severe: ~0.57
        class_centers = [0.43, 0.485, 0.515, 0.57]
        estimated_flood_prob = float(np.sum(probs * np.array(class_centers)))
        
        # Domain mitigation recommendations
        mitigation_actions = generate_mitigation_actions(input_dict)
        
        # Top 3 contributing adverse factors
        adverse_features = sorted(
            [(k, v) for k, v in input_dict.items() if k not in ["TopographyDrainage", "DrainageSystems", "DamsQuality", "RiverManagement", "Watersheds"]],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return jsonify({
            "status": "success",
            "prediction": {
                "risk_tier": pred_label,
                "flood_probability_score": round(estimated_flood_prob * 100, 2),
                "probabilities": prob_dict,
                "top_adverse_factors": adverse_features,
                "recommendations": mitigation_actions
            }
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(evaluation_metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

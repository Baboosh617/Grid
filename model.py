"""Loads the trained artifacts and runs inference, separate from the
Streamlit UI so it can be tested directly.
"""
import joblib
import pandas as pd

ARTIFACT_PATHS = {
    "model": "grid_stability_model.pkl",
    "scaler": "scaler.pkl",
    "label_encoder": "label_encoder.pkl",
    "feature_columns": "feature_columns.pkl",
}


def load_artifacts():
    model = joblib.load(ARTIFACT_PATHS["model"])
    scaler = joblib.load(ARTIFACT_PATHS["scaler"])
    le = joblib.load(ARTIFACT_PATHS["label_encoder"])
    feature_cols = joblib.load(ARTIFACT_PATHS["feature_columns"])
    return model, scaler, le, feature_cols


def predict(input_data: pd.DataFrame, model, scaler, le):
    """Returns (label, {class_name: probability})."""
    input_scaled = scaler.transform(input_data)
    pred_encoded = model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(input_scaled)[0]
    return pred_label, dict(zip(le.classes_, probs))

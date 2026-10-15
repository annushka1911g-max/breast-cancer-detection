import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer

# Cached paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_best.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature names from standard dataset
_dataset = load_breast_cancer()
FEATURE_NAMES = list(_dataset.feature_names)
# In Wisconsin Breast Cancer: 0 = Malignant, 1 = Benign
CLASS_MAPPING = {0: "Malignant", 1: "Benign"}

# Tree explainer for SHAP local interpretability
try:
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None


def _format_input_dataframe(features):
    """Ensure input is a DataFrame with proper column names."""
    if isinstance(features, pd.DataFrame):
        if list(features.columns) == FEATURE_NAMES:
            return features
        return pd.DataFrame(features.values, columns=FEATURE_NAMES)
    
    features_arr = np.array(features)
    if features_arr.ndim == 1:
        features_arr = features_arr.reshape(1, -1)
    return pd.DataFrame(features_arr, columns=FEATURE_NAMES)


def predict(features):
    """
    Predict tumor malignancy for a single patient sample.
    
    Returns:
        dict containing:
            - prediction: int (0: Malignant, 1: Benign)
            - label: str ("Malignant" or "Benign")
            - prob_malignant: float
            - prob_benign: float
            - confidence: float
            - risk_level: str ("Low", "Moderate", "High")
            - probabilities: list [prob_malignant, prob_benign]
    """
    df = _format_input_dataframe(features)
    scaled = scaler.transform(df)
    
    pred = int(model.predict(scaled)[0])
    probs = model.predict_proba(scaled)[0]
    
    # Class 0: Malignant, Class 1: Benign
    prob_malignant = float(probs[0])
    prob_benign = float(probs[1])
    
    # Risk categorization
    if prob_malignant >= 0.70:
        risk_level = "High Risk"
    elif prob_malignant >= 0.30:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"
        
    label = CLASS_MAPPING.get(pred, "Unknown")
    confidence = max(prob_malignant, prob_benign)
    
    return {
        "prediction": pred,
        "label": label,
        "prob_malignant": prob_malignant,
        "prob_benign": prob_benign,
        "confidence": confidence,
        "risk_level": risk_level,
        "probabilities": [prob_malignant, prob_benign]
    }


def batch_predict(df_features):
    """
    Predict for multiple patient records.
    Returns DataFrame with input data and added prediction columns.
    """
    df = _format_input_dataframe(df_features)
    scaled = scaler.transform(df)
    
    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)
    
    results = df.copy()
    results["Prediction"] = [CLASS_MAPPING[p] for p in preds]
    results["Malignant_Probability"] = np.round(probs[:, 0], 4)
    results["Benign_Probability"] = np.round(probs[:, 1], 4)
    results["Confidence"] = np.round(np.max(probs, axis=1) * 100, 2)
    results["Risk_Level"] = [
        "High Risk" if p >= 0.70 else ("Moderate Risk" if p >= 0.30 else "Low Risk")
        for p in probs[:, 0]
    ]
    return results


def explain_sample(features):
    """
    Compute local SHAP values for a sample.
    Returns top features driving towards malignant vs benign.
    """
    if explainer is None:
        return None
        
    df = _format_input_dataframe(features)
    scaled = scaler.transform(df)
    shap_values = explainer.shap_values(scaled)[0]
    
    # In XGBoost, higher log-odds indicate Class 1 (Benign), lower indicates Class 0 (Malignant)
    # So negative SHAP values increase malignancy risk
    contrib_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "value": df.values[0],
        "shap_value": shap_values,
        "impact": ["Benign Indicator" if s > 0 else "Malignant Indicator" for s in shap_values],
        "abs_importance": np.abs(shap_values)
    }).sort_values(by="abs_importance", ascending=False)
    
    return contrib_df
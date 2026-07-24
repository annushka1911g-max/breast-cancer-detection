import joblib
import numpy as np

# Load once when the module is imported
model = joblib.load("models/xgboost_best.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict(features):
    """
    features: list or numpy array of 30 feature values
    Returns:
        prediction (0/1),
        probabilities
    """
    features = np.array(features).reshape(1, -1)

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0]

    return prediction, probability
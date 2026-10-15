import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


def preprocess_data():
    """
    Loads Wisconsin Breast Cancer dataset, performs 80/20 stratified train/test split,
    applies StandardScaler and caches the fitted scaler.
    """
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X.columns
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = preprocess_data()
    print("Training Samples:", X_train.shape)
    print("Testing Samples:", X_test.shape)
    print("Features:", len(features))
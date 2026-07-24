import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd


def preprocess_data():

    # Load dataset
    dataset = load_breast_cancer()

    df = pd.DataFrame(dataset.data,
                      columns=dataset.feature_names)

    # Target
    y = dataset.target

    # Features
    X = df

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

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
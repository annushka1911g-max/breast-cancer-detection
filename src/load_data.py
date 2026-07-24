from sklearn.datasets import load_breast_cancer
import pandas as pd
def load_dataset():
    # Load dataset
    data = load_breast_cancer()

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add target column
    df["target"] = data.target

    # Convert numeric labels to readable labels
    df["diagnosis"] = df["target"].map({
        0: "Malignant",
        1: "Benign"
    })

    return df

if __name__ == "__main__":
    df = load_dataset()

    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nClass Distribution:")
    print(df["diagnosis"].value_counts())
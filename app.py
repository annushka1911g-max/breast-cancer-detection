import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from src.predict import predict
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Predict",
        "Model Performance",
        "About"
    ]
)
if page == "Home":

    st.title("🩺 Breast Cancer Detection")

    st.write("""
This application predicts whether a breast tumor is:

- Benign
- Malignant

using an XGBoost Machine Learning model trained on the Wisconsin Breast Cancer Dataset.
""")

    st.image("images/banner.png", use_container_width=True)
elif page == "Predict":

    st.title("Predict Breast Cancer")
data = load_breast_cancer()

feature_names = data.feature_names
user_input = []

for feature in feature_names:

    value = st.number_input(
        feature,
        value=0.0,
        format="%.4f"
    )

    user_input.append(value)
if st.button("Predict"):

    input_df = pd.DataFrame([user_input], columns=feature_names)
    prediction, probability = predict(input_df)
    confidence = np.max(probability)
    if prediction == 0:
         st.success("benign")

    else:
       st.error("malignant")
    st.write(confidence)
       


    st.metric(
    "Confidence",
    f"{confidence*100:.2f}%"
        )
prob_df = pd.DataFrame({

    "Class": ["Malignant", "Benign"],

    "Probability": probability
})

st.bar_chart(
    prob_df.set_index("Class")
)
if page == "Model Performance":
    st.title("Model Performance")
st.image("images/confusion_matrix.png")

st.image("images/roc_curve.png")

st.image("images/pr_curve.png")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "98.3%")

    st.metric("Precision", "98.8%")

with col2:
    st.metric("Recall", "99.1%")

    st.metric("F1 Score", "98.9%")
if page == "About":

    st.title("About")

    st.write("""
Developer:
Anushka Dwivedi

Technology Stack

• Python

• XGBoost

• Scikit-learn

• Streamlit

• SHAP

• Pandas

Dataset

Wisconsin Breast Cancer Dataset
""")

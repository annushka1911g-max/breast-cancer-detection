"""
Test suite for Breast Cancer Detection Platform.
Validates dependencies, model artifacts, inference pipeline, and XAI explainability.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.predict import predict, batch_predict, explain_sample, FEATURE_NAMES


class TestBreastCancerPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data = load_breast_cancer()
        cls.X = cls.data.data
        cls.y = cls.data.target
        # Class 0: Malignant, Class 1: Benign
        cls.sample_malignant = cls.X[0]
        cls.sample_benign = cls.X[19]

    def test_single_prediction_malignant(self):
        """Verify confirmed malignant sample is classified correctly."""
        res = predict(self.sample_malignant)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(res["label"], "Malignant")
        self.assertGreater(res["prob_malignant"], 0.80)
        self.assertEqual(res["risk_level"], "High Risk")

    def test_single_prediction_benign(self):
        """Verify confirmed benign sample is classified correctly."""
        res = predict(self.sample_benign)
        self.assertEqual(res["prediction"], 1)
        self.assertEqual(res["label"], "Benign")
        self.assertGreater(res["prob_benign"], 0.80)
        self.assertEqual(res["risk_level"], "Low Risk")

    def test_batch_prediction(self):
        """Verify batch prediction on multiple patient records."""
        df_test = pd.DataFrame(self.X[:5], columns=FEATURE_NAMES)
        res_df = batch_predict(df_test)
        self.assertEqual(len(res_df), 5)
        self.assertIn("Prediction", res_df.columns)
        self.assertIn("Malignant_Probability", res_df.columns)
        self.assertIn("Benign_Probability", res_df.columns)
        self.assertIn("Risk_Level", res_df.columns)

    def test_shap_explanation(self):
        """Verify local SHAP explanation computes without errors."""
        shap_df = explain_sample(self.sample_malignant)
        self.assertIsNotNone(shap_df)
        self.assertEqual(len(shap_df), len(FEATURE_NAMES))
        self.assertIn("shap_value", shap_df.columns)
        self.assertIn("impact", shap_df.columns)


if __name__ == "__main__":
    unittest.main()

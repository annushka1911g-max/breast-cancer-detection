import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import preprocess_data
model = joblib.load("models/xgboost_best.pkl")
X_train, X_test, y_train, y_test, feature_names = preprocess_data()
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]
print("Accuracy :", accuracy_score(y_test, predictions))

print("Precision:", precision_score(y_test, predictions))

print("Recall   :", recall_score(y_test, predictions))

print("F1 Score :", f1_score(y_test, predictions))

print("ROC AUC  :", roc_auc_score(y_test, probabilities))
print(classification_report(
    y_test,
    predictions
))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.savefig("images/confusion_matrix.png")

plt.show()
RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("ROC Curve")

plt.savefig("images/roc_curve.png")

plt.show()
PrecisionRecallDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Precision-Recall Curve")

plt.savefig("images/pr_curve.png")

plt.show()
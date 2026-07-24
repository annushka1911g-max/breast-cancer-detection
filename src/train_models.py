import joblib
import pandas as pd

from preprocess import preprocess_data

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
X_train, X_test, y_train, y_test, feature_names = preprocess_data()
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "Support Vector Machine":
        SVC(probability=True),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
}
results = []

best_model = None

best_score = 0
for name, model in models.items():

    print("=" * 60)

    print(name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if accuracy > best_score:

        best_score = accuracy

        best_model = model
joblib.dump(
    best_model,
    "models/model.pkl"
)
results_df = pd.DataFrame(results)

print(results_df)
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)
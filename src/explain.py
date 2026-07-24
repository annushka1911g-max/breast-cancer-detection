import joblib
import shap
import matplotlib.pyplot as plt

from preprocess import preprocess_data
model = joblib.load("models/xgboost_best.pkl")
X_train, X_test, y_train, y_test, feature_names = preprocess_data()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names
)
plt.savefig("images/shap_summary.png")
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="bar"
)
plt.savefig("images/shap_bar.png")
sample = 0
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample],
        base_values=explainer.expected_value,
        data=X_test[sample],
        feature_names=feature_names
    )
)
shap.initjs()

shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test[0],
    feature_names=feature_names
)
import matplotlib.pyplot as plt

def explain_prediction(sample_index):

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_index],
            base_values=explainer.expected_value,
            data=X_test[sample_index],
            feature_names=feature_names
        )
    )

    plt.show()
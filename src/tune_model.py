import joblib

from preprocess import preprocess_data

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
X_train, X_test, y_train, y_test, feature_names = preprocess_data()
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [3, 4, 5, 6],

    "learning_rate": [0.01, 0.05, 0.1, 0.2],

    "subsample": [0.8, 0.9, 1.0],

    "colsample_bytree": [0.8, 0.9, 1.0],

    "min_child_weight": [1, 3, 5],

    "gamma": [0, 0.1, 0.3]
}
search = RandomizedSearchCV(

    estimator=xgb,

    param_distributions=param_grid,

    n_iter=20,

    scoring="f1",

    cv=5,

    random_state=42,

    verbose=2,

    n_jobs=-1
)
search.fit(X_train, y_train)
print(search.best_params_)
best_model = search.best_estimator_

joblib.dump(
    best_model,
    "models/xgboost_best.pkl"
)
import time

import pandas as pd
import numpy as np
from scripts.utils import export_classification_reports, export_model
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
import joblib

# -----------------------------
# 1️⃣ Load preprocessed data
# -----------------------------
print("Loading preprocessed data")
X_train_val = joblib.load("./data/preprocessed/X_train_preprocessed.pkl")
X_test = joblib.load("./data/preprocessed/X_test_preprocessed.pkl")

y_train_val = joblib.load("./data/preprocessed/y_train_preprocessed.pkl").values.ravel()
y_test = joblib.load("./data/preprocessed/y_test_preprocessed.pkl").values.ravel()

# -----------------------------
# 2️⃣ Split train into train + validation for early stopping rounds on best estimator
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

# -----------------------------
# 3️⃣ Encode labels
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# -----------------------------
# 4️⃣ Create XGBoost model
# -----------------------------
xgb = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    n_jobs=1,
    early_stopping_rounds=10
)

# -----------------------------
# 5️⃣ Define hyperparameter distribution for HalvingGridSearchCV
# -----------------------------
param_list = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

# -----------------------------
# 6️⃣ HalvingGridSearchCV
# -----------------------------
print("Starting Grid Search")
search = HalvingGridSearchCV(
    estimator=xgb,
    param_grid=param_list,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1,
    factor=3,
    min_resources="exhaust",
    aggressive_elimination=False
)

# Fit grid search
start_time = time.time()
search.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)])

# -----------------------------
# 7️⃣ Best hyperparameters
# -----------------------------
best_params = search.best_params_
print("Best hyperparameters :", best_params)

# -----------------------------
# 8️⃣ Train final model
# -----------------------------
print("Training final model with best hyperparameters")

# Recreate the estimator because of HalvingGridSearch wrapper bug blocking kwargs on best_estimator_
best_estimator = search.best_estimator_
best_estimator.fit(
    X_train,
    y_train_enc,
    eval_set=[(X_val, y_val_enc)]
)

end_time = time.time()
elapsed = end_time - start_time
elapsed_formatted = f"Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)"
print(elapsed_formatted)

# -----------------------------
# 9️⃣ Evaluate on test set and save results
# -----------------------------
print("Evaluating on test set")
y_pred_enc = best_estimator.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

export_model('xgboost', best_estimator)
export_classification_reports('xgboost', y_pred, y_test, best_params, param_list, elapsed_formatted)
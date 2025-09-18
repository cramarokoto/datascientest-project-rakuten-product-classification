import time

import pandas as pd
import numpy as np

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils import export_classification_reports, export_model, load_preprocessed_text_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------
# Load preprocessed data
# -----------------------------
X_train, X_test, y_train, y_test = load_preprocessed_text_data()
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# -----------------------------
# Create Logistic Regression model and define hyperparameters
# -----------------------------
lr = LogisticRegression(max_iter=1000)

param_list = {
    'solver': ["lbfgs", "saga"],
    'C': [0.01, 0.1, 1, 10, 100],
}

# -----------------------------
# GridSearchCV
# -----------------------------
print("Starting Grid Search")
search = GridSearchCV(
    estimator=lr,
    param_grid=param_list,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1,
    verbose=2,
)

start_time = time.time()
search.fit(X_train, y_train)

# -----------------------------
# Best hyperparameters
# -----------------------------
best_params = search.best_params_
print("Best hyperparameters :", best_params)

# -----------------------------
# Train final model
# -----------------------------
print("Training final model with best hyperparameters")
best_estimator = search.best_estimator_
best_estimator.fit(
    X_train,
    y_train,
)

end_time = time.time()
elapsed = end_time - start_time
elapsed_formatted = f"Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)"
print(elapsed_formatted)

# -----------------------------
# Evaluate on test set
# -----------------------------
print("Evaluating on test set")
y_pred = best_estimator.predict(X_test)


# ----------------------------
# Save the model and classification report
# -----------------------------

export_model('logistic_regression_text', best_estimator)
export_classification_reports('logistic_regression_text', y_pred, y_test, best_params, param_list, elapsed_formatted)

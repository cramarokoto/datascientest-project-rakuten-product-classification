import time

import pandas as pd
import numpy as np

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils import export_classification_reports, export_model, load_preprocessed_text_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------
# Load preprocessed data
# -----------------------------
X_train = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'tfidf/X_train_tfidf.pkl'))

X_test = joblib.load(os.path.join(PROJECT_ROOT, 'data', 'tfidf/X_test_tfidf.pkl'))

y_train = joblib.load(os.path.join(PROJECT_ROOT, 'data', "tfidf/y_train_tfidf.pkl")).values.ravel()
y_test = joblib.load(os.path.join(PROJECT_ROOT, 'data', "tfidf/y_test_tfidf.pkl")).values.ravel()

# -----------------------------
# Create Random Forest model and define hyperparameters
# -----------------------------
rfc = RandomForestClassifier(random_state=42)

param_list = {
    'n_estimators': [50, 100, 200], 
    'max_depth': [10, 20, 30]
}

# -----------------------------
# RandomizedSearchCV
# -----------------------------
print("Starting Randomized Search")
search = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_list,
    n_iter=6,
    scoring='f1_weighted',
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
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


# -----------------------------
# Save the model and classification report
# -----------------------------

export_model('random_forest_tfidf', best_estimator)
export_classification_reports('random_forest_tfidf', y_pred, y_test, best_params, param_list, elapsed_formatted)

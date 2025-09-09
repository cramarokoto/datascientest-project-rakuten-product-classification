import time

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------
# Load preprocessed data
# -----------------------------
print("Loading preprocessed data")
X_train = joblib.load("../data/preprocessed/X_train_preprocessed.pkl")
X_test = joblib.load("../data/preprocessed/X_test_preprocessed.pkl")

y_train = joblib.load("../data/preprocessed/y_train_preprocessed.pkl").values.ravel()
y_test = joblib.load("../data/preprocessed/y_test_preprocessed.pkl").values.ravel()


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
print(f"Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)")

# -----------------------------
# Evaluate on test set
# -----------------------------
print("Evaluating on test set")
y_pred = best_estimator.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Save the model and classification report
# -----------------------------
print("Saving model and classification report")
joblib.dump(best_estimator, '../models/random_forest_model.pkl')

with open('../models/random_forest_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))
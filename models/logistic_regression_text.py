import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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
joblib.dump(best_estimator, '../models/logistic_regression_text_model.pkl')

with open('../models/logistic_regression_text_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib

X_train = joblib.load("../data/preprocessed/X_train_preprocessed.pkl")
X_test = joblib.load("../data/preprocessed/X_test_preprocessed.pkl")

y_train = joblib.load("../data/preprocessed/y_train_preprocessed.pkl").values.ravel()
y_test = joblib.load("../data/preprocessed/y_test_preprocessed.pkl").values.ravel()

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Création du model XGBoost
xgb = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    n_jobs=1
)

# Définition des hyperparamètres pour la recherche par grille
param_grid = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Recherche par grille avec validation croisée
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3,
    n_jobs=7,
    verbose=2
)
grid_search.fit(X_train, y_train_enc)

# Meilleurs hyperparamètres
best_params = grid_search.best_params_
print("Meilleurs hyperparamètres :", best_params)

# Entraînement du modèle avec les meilleurs hyperparamètres
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train_enc)

# Prédictions sur le jeu de test
y_pred_enc = best_xgb.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)
report = classification_report(y_test, y_pred, output_dict=True)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Sauvegarde du modèle
joblib.dump(best_xgb, './models/xgboost_model.pkl')

# Sauvegarde du classification report en fichier txt
with open('./models/xgboost_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))

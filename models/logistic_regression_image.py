import time
import joblib
import numpy as np
from PIL import Image

from scripts.utils import load_sampled_paths_data, encoding_labels, export_classification_reports, export_model

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Batching images and Reduce dimensionality with Incremental PCA 
# -----------------------------
# We have 200x200 images in grayscale, so 40000 features.
# We cannot load all images in memory at once, so we need to read them in batches.
# We will use IncrementalPCA to reduce the dimensionality of the images before feeding them to the Logistic Regression.
# This will highly reduce the training time and the memory usage.

def iter_batches_from_disk(df, batch_size=512):
    """
    Generator that reads images per batch from df['processed_image_path'].
    Works with 200x200 images in grayscale.
    Return an array (batch_size, 40000).
    """
    for start in range(0, len(df), batch_size):
        batch_paths = df.iloc[start:start+batch_size]["processed_image_path"].values
        X_batch = []
        for p in batch_paths:
            img = Image.open(p)
            arr = np.array(img, dtype=np.float32) / 255.0  # normalisation
            X_batch.append(arr.flatten())                  # (40000,)
        yield np.stack(X_batch)


def fit_incremental_pca(train_df, n_components=256, batch_size=512):
    start_time = time.time()
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Learn components in 1st pass
    for X_batch in iter_batches_from_disk(train_df, batch_size):
        ipca.partial_fit(X_batch)
    
    # Timer
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time for IncrementalPCA fitting: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")

    print("IncrementalPCA fitted with n_components =", n_components)
    joblib.dump(ipca, "./models/ipca_for_image_logreg.pkl")
    print("IncrementalPCA model saved.")

    return ipca


def reduce_dimensionality(ipca, df, batch_size=512):
    start_time = time.time()
    # Transform data in 2nd pass
    X_red = []
    for X_batch in iter_batches_from_disk(df, batch_size):
        X_red.append(ipca.transform(X_batch))
    X_red = np.vstack(X_red)

    # Timer
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time to reduce dimension: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")

    return X_red


# -----------------------------
# Grid Search Halving
# -----------------------------
def halving_logreg(X_train_red, y_train_enc, random_state=42):
    logreg = LogisticRegression(max_iter=1000, random_state=random_state)

    param_grid = [
        # lbfgs + L2
        {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"],
            "penalty": ["l2"]
        },
        # saga + L1/ElasticNet
        {
            "C": [0.1, 1, 10],
            "solver": ["saga"],
            "penalty": ["l1", "elasticnet"],
            "l1_ratio": [0.5]
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = HalvingGridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        aggressive_elimination=True,
        n_jobs=-1,
        verbose=2,
        refit=True,
        random_state=random_state
    )

    start_time = time.time()
    # Training  the model with the reduced data
    search.fit(X_train_red, y_train_enc)

    # Timer
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time for the search and training: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")

    return search


def save_model_and_results(search_model, X_test_red, y_test_enc, le):
    # Best hyperparameters
    best_params = search_model.best_params_
    print("Best hyperparameters :", best_params)

    # Save the model
    best_estimator = search_model.best_estimator_
    print("Saving model")
    joblib.dump(best_estimator, './models/logistic_regression_image_model.pkl')

    print("Evaluating on test set")
    y_pred_enc = search_model.predict(X_test_red)

    y_pred = le.inverse_transform(y_pred_enc)
    y_test = le.inverse_transform(y_test_enc)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", report)
    with open('./models/logistic_regression_image_report.txt', 'w') as f:
        f.write(report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    with open('./models/logistic_regression_image_confusion_matrix.txt', 'w') as f:
        f.write(conf_matrix)

    return report, conf_matrix
    
# -----------------------------
# Main & Tests
# -----------------------------
def whole_process_and_training():
    # Prepare data
    print("### Preparing data")
    train_sampled_df, test_df = load_sampled_paths_data()
    y_train_enc, y_test_enc, label_map, inverse_map, le = encoding_labels(train_sampled_df, test_df)
    print("Encoded y shapes:", y_train_enc.shape, y_test_enc.shape)

    # Fit IncrementalPCA and reduce dimensionality
    print("### Fitting IncrementalPCA and reducing dimensionality")
    
    ipca = fit_incremental_pca(train_sampled_df, n_components=256, batch_size=512)
    X_train_red = reduce_dimensionality(ipca, train_sampled_df, batch_size=512)
    joblib.dump(X_train_red, './data/images/X_train_reduced_image_logreg.pkl')
    print("Reduced X train shape:", X_train_red.shape)

    X_train_red = joblib.load('./data/images/X_train_reduced_image_logreg.pkl')
    X_test_red = reduce_dimensionality(ipca, test_df, batch_size=512)
    joblib.dump(X_test_red, './data/images/X_test_reduced_image_logreg.pkl')
    print("Reduced X_test shape:", X_test_red.shape)

    # Grid Search with Halving
    print("### Starting Halving Grid Search")
    search_model = halving_logreg(X_train_red, y_train_enc, random_state=42)

    # Save model and results
    print("### Saving model and results")

    report, conf_matrix = save_model_and_results(search_model, X_test_red, y_test_enc, le)

    # If everything worked, we can save everything in one file
    joblib.dump({
        "label_encoder": le,
        "ipca": ipca,
        "model": search_model.best_estimator_,
        "report": report,
        "confustion_matrix": conf_matrix,
        "fit_time": search_model.refit_time_
    }, './models/logistic_regression_image_full_data.pkl')

    print("\nMy job here is done.")
    return


def process_if_already_done():
    train_sampled_df, test_df = load_sampled_paths_data()

    full_data = joblib.load("./models/logistic_regression_image_full_data.pkl")

    le = full_data["label_encoder"]
    y_train_enc = le.transform(train_sampled_df["prdtypecode"])
    y_test_enc = le.transform(test_df["prdtypecode"])

    ipca = joblib.load("./models/ipca_for_image_logreg.pkl")

    X_train_red = joblib.load('./data/images/X_train_reduced_image_logreg.pkl') # To have locally
    X_test_red = joblib.load('./data/images/X_test_reduced_image_logreg.pkl') # To have locally

    best_model = full_data["model"]
    print("Best model:", best_model)

    start_time = time.time()
    best_model.fit(X_train_red, y_train_enc)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time to re-train the best model: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")
    
    joblib.dump(best_model, './models/logistic_regression_image_model.pkl')

    y_pred_enc = best_model.predict(X_test_red)
    y_pred = le.inverse_transform(y_pred_enc)
    y_test = le.inverse_transform(y_test_enc)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    with open("./models/logistic_regression_image_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    with open('./models/logistic_regression_image_confusion_matrix.txt', 'w') as f:
        f.write(str(conf_matrix))

    ipca = full_data["ipca"]

    joblib.dump({
        "label_encoder": le,
        "ipca": ipca,
        "model": best_model,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "fit_time": elapsed
    }, './models/logistic_regression_image_full_data.pkl')

    print("\nMy job here is done.")


if __name__ == "__main__":
    # whole_process_and_training() # First run
    # process_if_already_done()  # Uncomment it's not the first run
    full_data = joblib.load("./models/logistic_regression_image_full_data.pkl")

    report = full_data["classification_report"]
    print("Classification Report:\n", report)
    with open("./models/logistic_regression_image_report.txt", "w") as f:
        f.write(str(report))

    conf_matrix = full_data["confusion_matrix"]
    print("Confusion Matrix:\n", conf_matrix)
    with open('./models/logistic_regression_image_confusion_matrix.txt', 'w') as f:
        f.write(str(conf_matrix))
    pass
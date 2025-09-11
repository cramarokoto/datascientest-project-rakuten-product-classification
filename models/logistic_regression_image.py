import time
import joblib
import numpy as np
from PIL import Image

from scripts.utils import load_data, dataset_sampler_under_oversampling, prepare_images_df

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Load sampled data
# -----------------------------
def load_sampled_paths_data():
    print("Loading data (x, y)")
    X, y, _ = load_data()
    X_sampled, y_sampled = dataset_sampler_under_oversampling(X, y)
    X_sampled = prepare_images_df(X_sampled, n_images=None, split="train", random_state=42)

    df = X_sampled.merge(y_sampled, left_index=True, right_index=True)[["processed_image_path", "prdtypecode"]]

    return X_sampled, y_sampled, df


# -----------------------------
# Split train/test
# -----------------------------
def splitting_data(df, test_size=0.2, random_state=42):
    print("Splitting data into train/test")
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["prdtypecode"]
    )

    return train_df, test_df

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
# Encode labels
# -----------------------------
def encoding_labels(train_df, test_df):
    print("Encoding labels")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df["prdtypecode"])
    y_test_enc = le.transform(test_df["prdtypecode"])

    # dictionnaire prdtypecode -> index
    label_map = {cls: idx for idx, cls in enumerate(le.classes_)}
    # dictionnaire index -> prdtypecode
    inverse_map = {v: k for k, v in label_map.items()}

    
    return y_train_enc, y_test_enc, label_map, inverse_map, le


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


def save_model_and_results(search_model, X_test_red, y_test_enc):
    # Best hyperparameters
    best_params = search_model.best_params_
    print("Best hyperparameters :", best_params)

    # Save the model
    best_estimator = search_model.best_estimator_
    print("Saving model")
    joblib.dump(best_estimator, './models/logistic_regression_image_model.pkl')

    print("Evaluating on test set")
    y_pred = search_model.predict(X_test_red)

    report = classification_report(y_test_enc, y_pred, output_dict=True)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_pred))

    with open('./models/logistic_regression_image_report.txt', 'w') as f:
        f.write(classification_report(y_test_enc, y_pred))

    return report
    
# -----------------------------
# Main & Tests
# -----------------------------
def whole_process_and_training():
    # Prepare data
    print("### Preparing data")
    X_sampled, y_sampled, df = load_sampled_paths_data()
    train_df, test_df = splitting_data(df)
    y_train_enc, y_test_enc, label_map, inverse_map, le = encoding_labels(train_df, test_df)
    joblib.dump(le, './models/labelEncoder_image_logreg.pkl')
    print("Encoded y shapes:", y_train_enc.shape, y_test_enc.shape)

    # Fit IncrementalPCA and reduce dimensionality
    print("### Fitting IncrementalPCA and reducing dimensionality")
    ipca = fit_incremental_pca(train_df, n_components=256, batch_size=512)
    X_train_red = reduce_dimensionality(ipca, train_df, batch_size=512)
    print("Reduced X train shape:", X_train_red.shape)
    X_test_red = reduce_dimensionality(ipca, test_df, batch_size=512)
    print("Reduced X_test shape:", X_test_red.shape)

    # If you have done the preparation aldreay, comment before and uncomment the next line
    # X_sampled, y_sampled, df = load_sampled_paths_data()
    # train_df, test_df = splitting_data(df)
    # le = joblib.load('./models/labelEncoder_image_logreg.pkl')
    # y_train_enc = le.transform(train_df["prdtypecode"])
    # y_test_enc = le.transform(test_df["prdtypecode"])
    # ipca = joblib.load('./models/ipca_for_image_logreg.pkl')
    # X_train_red = reduce_dimensionality(ipca, train_df, batch_size=512)
    # X_test_red = reduce_dimensionality(ipca, test_df, batch_size=512)

    # Grid Search with Halving
    print("### Starting Halving Grid Search")
    search_model = halving_logreg(X_train_red, y_train_enc, random_state=42)

    # Save model and results
    print("### Saving model and results")
    report = save_model_and_results(search_model, X_test_red, y_test_enc)

    # If everything worked, we can save everything in one file
    joblib.dump({
        "label_encoder": le,
        "ipca": ipca,
        "model": search_model.best_estimator_,
        "report": report
    }, './models/logistic_regression_image_full_data.pkl')

    print("\nMy job here is done.")

    # just have to jump.load the preprocessed data next time


if __name__ == "__main__":
    
    X_sampled, y_sampled, df = load_sampled_paths_data()
    train_df, test_df = splitting_data(df)

    full_data = joblib.load("./models/logistic_regression_image_full_data.pkl")

    le = full_data["label_encoder"]
    y_train_enc = le.fit(train_df["prdtypecode"])
    y_test_enc = le.transform(test_df["prdtypecode"])

    X_train_red = full_data["X_train_red"]
    X_test_red = full_data["X_test_red"]
    # X_train_red = reduce_dimensionality(ipca, train_df, batch_size=512)
    # X_test_red = reduce_dimensionality(ipca, test_df, batch_size=512)

    best_model = full_data["model"]
    print("Best model:", best_model)

    start_time = time.time()
    best_model.fit(X_train_red, y_train_enc)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time to re-train the best model: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")
    
    joblib.dump(best_model, './models/logistic_regression_image_model.pkl')

    y_pred = best_model.predict(X_test_red)

    report = classification_report(y_test_enc, y_pred, output_dict=True)
    print("Classification Report:\n", report)
    with open('./models/logistic_regression_image_report.txt', 'w') as f:
        f.write(classification_report(y_test_enc, y_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_pred))

    ipca = full_data["ipca"]

    joblib.dump({
        "label_encoder": le,
        "ipca": ipca,
        "model": best_model,
        "report": report,
        "fit_time": elapsed
    }, './models/logistic_regression_image_full_data.pkl')

    print("\nMy job here is done.")
    


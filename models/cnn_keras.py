import time

import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

from scripts.utils import load_sampled_paths_data, encoding_labels

# -----------------------------
# Global variables
# -----------------------------
preprocessed_format = {
    "dim": (200, 200),
    "grayscale": True,
    "to_resize": True,
    "col_path": "processed_image_path",
}

brut_format = {
    "dim": (500, 500),
    "grayscale": False,
    "to_resize": True,
    "col_path": "image_path",
}

CURRENT_FORMAT = preprocessed_format
RESIZE_DIM = (224, 224)


# -----------------------------
# Prepare data
# -----------------------------
def train_paths_labels():
    print("### Preparing data")
    train_sampled_df, test_df = load_sampled_paths_data()
    y_train_enc, y_test_enc, label_map, inverse_map, le = encoding_labels(
        train_sampled_df, test_df
    )
    print("Encoded y shapes:", y_train_enc.shape, y_test_enc.shape)

    train_paths = train_sampled_df[CURRENT_FORMAT["col_path"]].values
    train_labels = y_train_enc

    return train_paths, train_labels, test_df, y_test_enc, le


# -----------------------------
# Reading and preprocess
# -----------------------------
def process_image(path, label):
    # lecture du fichier
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1 if CURRENT_FORMAT["grayscale"] else 3)

    # resize
    if CURRENT_FORMAT["to_resize"]:
        img = tf.image.resize(img, RESIZE_DIM)

    # normalisation
    img = img / 255.0

    return img, label


# -----------------------------
# Model CNN
# -----------------------------
def easy_cnn(num_classes):
    img_size = RESIZE_DIM if CURRENT_FORMAT["to_resize"] else CURRENT_FORMAT["dim"]
    model = models.Sequential(
        [
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                input_shape=img_size + (1 if CURRENT_FORMAT["grayscale"] else 3,),
            ),
            layers.MaxPooling2D(2),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2),
            
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# -----------------------------
# Model CNN
# -----------------------------
def make_test_dataset(test_df, y_test_enc):
    test_paths = test_df[CURRENT_FORMAT["col_path"]].values
    test_labels = y_test_enc

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return test_ds


# -----------------------------
# Main & Tests
# -----------------------------
def main_cnn():  # Modify CURRENT_FORMAT or RESIZE_DIM to try other parameters.
    start_time = time.time()

    train_paths, train_labels, test_df, y_test_enc, le = (
        train_paths_labels()
    )

    num_classes = len(le.classes_)
    print("\nle classes encodés:", le.classes_)

    dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    val_split = 0.2
    ds_size = len(dataset)
    val_size = int(ds_size * val_split)

    val_ds = dataset.take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
    train_ds = dataset.skip(val_size).batch(32).prefetch(tf.data.AUTOTUNE)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time to prepare all data for the model: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")

    model = easy_cnn(num_classes)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    start_time = time.time()

    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time to train the model: {elapsed:.2f} seconds, or {elapsed/60:.2f} minutes, or {elapsed/3600:.2f} hours")

    model.save("models/cnn_model.keras")

    test_ds = make_test_dataset(test_df, y_test_enc)

    # Évaluation
    loss, acc = model.evaluate(test_ds)
    print(f"✅ Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")

    y_true_enc = []
    y_pred_enc = []

    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        y_true_enc.extend(labels.numpy())
        y_pred_enc.extend(np.argmax(preds, axis=1))
    
    # ----- Debug infos -----
    y_true_enc = np.array(y_true_enc)
    y_pred_enc = np.array(y_pred_enc)

    print("👉 Nombre d'échantillons test :", len(y_true_enc))
    print("👉 Labels uniques dans le test set :", np.unique(y_true_enc))
    print("👉 Labels uniques prédits :", np.unique(y_pred_enc))

    print("\nDistribution des vraies classes (test) :")
    print(pd.Series(y_true_enc).value_counts().sort_index())

    print("\nDistribution des classes prédites :")
    print(pd.Series(y_pred_enc).value_counts().sort_index())
    # ----- ----------- -----

    y_test = le.inverse_transform(y_true_enc)
    y_pred = le.inverse_transform(y_pred_enc)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", str(report))
    with open("./models/cnn_image_classification_report.txt", "w") as f:
        f.write(str(report))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", str(conf_matrix))
    with open('./models/cnn_image_confusion_matrix.txt', 'w') as f:
        f.write(str(conf_matrix))

    joblib.dump(
        {
            "model": model,
            "le": le,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "fit_time": elapsed,
        },
        "./models/cnn_image_full_data.pkl",
    )

    with open("cnn_image_history.json", "w") as f:
        json.dump(history.history, f)

    print("\nMy job here is done.")


def if_model_saved():

    train_paths, train_labels, test_df, y_test_enc, le = (
        train_paths_labels()
    )

    model = load_model("models/cnn_model.keras")
    # model = load_model("models/cnn_model.h5") # Depends on the one saved

    test_ds = make_test_dataset(test_df, y_test_enc)

    # Évaluation
    loss, acc = model.evaluate(test_ds)
    print(f"✅ Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")

    y_true_enc = []
    y_pred_enc = []

    for imgs, labels in test_ds:
        preds = model.predict(imgs)
        y_true_enc.extend(labels.numpy())
        y_pred_enc.extend(np.argmax(preds, axis=1))
    
    # ----- Debug infos -----
    y_true_enc = np.array(y_true_enc)
    y_pred_enc = np.array(y_pred_enc)

    print("👉 Nombre d'échantillons test :", len(y_true_enc))
    print("👉 Labels uniques dans le test set :", np.unique(y_true_enc))
    print("👉 Labels uniques prédits :", np.unique(y_pred_enc))

    print("\nDistribution des vraies classes (test) :")
    print(pd.Series(y_true_enc).value_counts().sort_index())

    print("\nDistribution des classes prédites :")
    print(pd.Series(y_pred_enc).value_counts().sort_index())
    # ----- ----------- -----

    y_test = le.inverse_transform(y_true_enc)
    y_pred = le.inverse_transform(y_pred_enc)

    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    with open("./models/cnn_image_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    with open('./models/cnn_image_confusion_matrix.txt', 'w') as f:
        f.write(str(conf_matrix))

    full_data = joblib.load("./models/cnn_image_full_data.pkl")
    elapsed = full_data["fit_time"]

    joblib.dump(
        {
            "model": model,
            "le": le,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "fit_time": elapsed,
        },
        "./models/cnn_image_full_data.pkl",
    )

    print("\nMy job here is done.")



if __name__ == "__main__":
    # main_cnn() # If first time
    # if_model_saved() # For other check
    pass
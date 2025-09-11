import os

# Models or object serialization
import joblib

# Data analysis libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# IPython Notebook magic
from IPython.display import display

# Data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Image processing library
from PIL import Image

# Sampling libraries
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Chargement des données
def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_update.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(DATA_DIR, "Y_train_CVw08PX.csv"), index_col=0)
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_update.csv"), index_col=0)
    return X_train, y_train, X_test

X_train, y_train, X_test = load_data()

# Infos de bases des dataframes
def data_info(df):
    df.info()
    print("\n")
    display(df.head())
    print("\n")


#####################################
############ IMAGE UTILS ############
#####################################

# Paths
image_train_path = os.path.join(DATA_DIR, "images/image_train/")
image_test_path = os.path.join(DATA_DIR, "images/image_test/")
image_gray_path_train = os.path.join(DATA_DIR, "images/gray_resized_images_train/")
image_gray_path_test = os.path.join(DATA_DIR, "images/gray_resized_images_test/")

#### Displaying of images

# Find a picture path from its imageid and its productid
def image_path(imageid, productid, split="train"):
    if split == "train":
        path = image_train_path
    elif split == "test":
        path = image_test_path
    else:
        raise Exception("split must be train or test.")

    image_path = path + "image_" + str(imageid) + "_product_" + str(productid) + ".jpg"
    return image_path


def find_image_gray_path(imageid, productid, split="train"):
    if split == "train":
        path = image_gray_path_train
    elif split == "test":
        path = image_gray_path_test
    else:
        raise Exception("split must be train or test.")
    
    image_path = path + "image_" + str(imageid) + "_product_" + str(productid) + ".jpg"
    return image_path


# Display an image from its path or an image objetc
def display_image(image, print_dim=False, cmap='viridis'):
    if isinstance(image, str):
        img = mpimg.imread(image)
    else:
        img = image

    if print_dim:
        print(f"Image dimensions: {img.shape[0]}x{img.shape[1]} pixels")

    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()
    return


# Display an image from its imageid and its productid
def display_image_df(imageid, productid, split="train", print_dim=False):
    display_image(image_path(imageid, productid, split), print_dim)
    return


# Display an image from a textual row
def display_image_from_row(index_or_row_number, split="train", is_index=True, print_dim=False):
    """
    Display a picture from index (is_index must be True) or
    row number (is_index must be False) of the considered split.
    print_dim allows to print the dimensions of the image.
    """
    if split == "train":
        df = X_train
    elif split == "test":
        df = X_test
    else:
        raise Exception("split must be train or test.")
        
    if is_index:
        display_image_df(df.loc[index_or_row_number, "imageid"], df.loc[index_or_row_number, "productid"], split, print_dim)
    else:
        display_image_df(df.iloc[index_or_row_number, 3], df.iloc[index_or_row_number, 2], split, print_dim)
    return

#### Images format
def shape_from_path(image_path):
    img = mpimg.imread(image_path)
    return img.shape

#### Content Box

def get_content_box(image_path, seuil_ratio=0.99):
    """
    Détecte la bounding box du contenu non blanc d'une image.

    Args:
        image_path (str): chemin vers l'image.
        seuil (float): seuil pour considérer un pixel blanc (entre 0 et 1).
    Returns:
        tuple: dimensions de la zone utile (largeur, hauteur), coordonnées x_min, y_min, x_max, y_max.
    """
     
    # Charge l'image (forme: H x W x C), valeurs normalisées [0, 1]
    img = mpimg.imread(image_path)

    # img peut être chargé normalisé (valeurs de couleurs entre 0 et 1) ou en [0, 255]
    # On adapte donc le seuil en fonction de la valeur maximale des pixels
    seuil = seuil_ratio * img.max()

    # Crée un masque des pixels "non blancs"
    # Ici, on considère qu'un pixel est blanc si ses 3 canaux sont > seuil (proche de 1)
    non_white = np.any(img < seuil, axis=2)
   
    # Trouve les coordonnées de la zone utile
    coords = np.argwhere(non_white)

    if coords.size == 0:
        # "Image entièrement blanche."
        return None

    # Récupère les limites du contenu utile
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # print(f"Taille utile : {width} x {height} pixels")
    # print(f"Bounding box : x={x_min}, y={y_min}, largeur={width}, hauteur={height}")

    return ((width, height), x_min, y_min, x_max, y_max)

# Charge l'image complète et retourne l'image utile à partir de la bounding box
def load_content_from_box(image_path, x_min, y_min, x_max, y_max):
    # Charge l'image complète
    img = mpimg.imread(image_path)

    # Attention : dans numpy, l'ordre est img[hauteur, largeur] donc img[y_min:y_max+1, x_min:x_max+1]
    img_cropped = img[y_min:y_max+1, x_min:x_max+1, :]
    return img_cropped


def load_content_box_from_image(image_path):
    """
    Charge l'image complète et retourne l'image utile.
    
    Args:
        image_path (str): chemin vers l'image.
    
    Returns:
        np.ndarray: image utile sans les bords blancs.
    """
    dim, x_min, y_min, x_max, y_max = get_content_box(image_path)
    if dim is None:
        return None  # Image entièrement blanche
    
    return load_content_from_box(image_path, x_min, y_min, x_max, y_max)

#### Autres

# Conversion en niveaux de gris
def to_grayscale(img_path, normalize=False):
    img = Image.open(img_path).convert("L")
    # Passage en float32 pour diviser par 2 la mémoire sans perte notable de précision
    if normalize:
        img = np.asarray(img, dtype=np.float32) / 255.0  # Normalisation entre 0 et 1
    return img


def prepare_images_df(df_X, n_images=None, split="train", random_state=42):
    """
    Prepare a dataframe with a sample of n_images and their image paths.
    """
    if n_images is None:
        X_img_sample = df_X.copy()
    else:
        X_img_sample = df_X.sample(n=n_images, random_state=random_state)

    X_img_sample["image_path"] = X_img_sample.apply(lambda row: image_path(row["imageid"], row["productid"], split), axis=1)
    X_img_sample["processed_image_path"] = X_img_sample.apply(lambda row: find_image_gray_path(row["imageid"], row["productid"], split), axis=1)
    X_img_sample = X_img_sample[["image_path", "processed_image_path"]]
    return X_img_sample


def adding_content_box(df_sample):
    """
    Add content box information to a dataframe with image paths.
    """
    df_content_box = df_sample["image_path"].apply(get_content_box).apply(pd.Series)
    df_content_box.columns = ["content_dim", "x_min", "y_min", "x_max", "y_max"]

    df_sample["content_dim"] = df_content_box["content_dim"]
    df_sample[["content_width", "content_height"]] = df_sample["content_dim"].apply(pd.Series)
    df_sample[["x_min", "y_min", "x_max", "y_max"]] = df_content_box[["x_min", "y_min", "x_max", "y_max"]]
    
    df_sample["content_ratio"] = np.round(df_sample["content_width"] / df_sample["content_height"], 4)

    return df_sample


#####################################
############# SAMPLING ##############
#####################################


def dataset_sampler_under_oversampling(X_train, y_train):
    # print("Train before over and undersampling :")
    # print(y_train['prdtypecode'].value_counts(normalize=True) * 100)

    # Calcul des effectifs
    counts = Counter(y_train['prdtypecode'])
    n_total = len(y_train)
    target_ratio = 0.06

    # Construction d'une sampling_strategy d'over et undersampling
    undersampling_strategy = {
        2583: int(n_total * target_ratio)
    }
    oversampling_strategy = {}
    for cls, count in counts.items():
        current_ratio = count / n_total
        if current_ratio < target_ratio:
            oversampling_strategy[cls] = int(n_total * target_ratio)

    # Application de l'over et undersampling avec un pipeline
    pipeline = Pipeline(steps=[
        ('under', RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)),
        ('over', RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=42))
    ])

    X_train, y_train = pipeline.fit_resample(X_train, y_train)

    # print("\nTrain after over and undersampling :")
    # print(y_train['prdtypecode'].value_counts(normalize=True) * 100)
    return X_train, y_train

#####################################
############# EXPORTS ###############
#####################################

def export_classification_reports(model_name, y_pred, y_test, best_params, search_params, elapsed_formatted):
    """
    Saves classification results to the './models/' directory.

    This function exports the following files:
        - '{model_name}_classification_report.txt': Text classification report comparing y_test and y_pred.
        - '{model_name}_confusion_matrix.txt': Confusion matrix comparing y_test and y_pred.
        - '{model_name}_training_info.txt': Training information including execution time and search/best parameters.

    Args:
        model_name (str): The name to use for the saved files.
        y_pred (array-like): Predicted labels from the classifier.
        y_test (array-like): True labels for the test set.
        best_params (dict or str): Best hyperparameters found during model selection.
        search_params (dict or str): Search hyperparameters used during model selection.
        elapsed_formatted (str): Formatted string representing the training execution time.

    Returns:
        None
    """
    print("Saving classification report")
    with open(f'./models/{model_name}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))

    with open(f'./models/{model_name}_confusion_matrix.txt', 'w') as f:
        f.write(str(confusion_matrix(y_test, y_pred)))

    with open(f'./models/{model_name}_training_info.txt', 'w') as f:
        f.write('Execution time : ' + str(elapsed_formatted) + '\n')
        f.write('Best params : ' + str(best_params) + '\n')
        f.write('Search params : ' + str(search_params) + '\n')

def export_model(model, model_name):
    """
    Exports the given model to a file using joblib.

    Args:
        model: The trained model object to be saved.
        model_name (str): The name to use for the saved model file.

    Returns:
        None

    Side Effects:
        Saves the model to './models/{model_name}_model.pkl'.

    """
    print("Saving model")
    joblib.dump(model, './models/#{model_name}_model.pkl')
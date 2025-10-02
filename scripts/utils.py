import os

# Models or object serialization
import joblib

# Data analysis libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample

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

# Text processing
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def load_data():
    """
    Loads Rakuten data challenge training and test datasets from raw CSV files located in the DATA_DIR directory.

    Returns:
        tuple:
            - X_train (pd.DataFrame): Features for the training set.
            - y_train (pd.DataFrame): Target labels for the training set.
            - X_test (pd.DataFrame): Features for the challenge test set without labels.

    Note:
        Assumes that DATA_DIR is defined and that the required CSV files exist in this directory.
    """
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_update.csv"), index_col=0)
    y_train = pd.read_csv(os.path.join(DATA_DIR, "Y_train_CVw08PX.csv"), index_col=0)
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test_update.csv"), index_col=0)
    return X_train, y_train, X_test

X_train, y_train, X_test = load_data()

def load_preprocessed_text_data():
    """
    Loads preprocessed training and test datasets for text classification.

    Returns:
        tuple: A tuple containing four elements:
            - X_train_val (csr matrix): Preprocessed features for training and validation.
            - X_test (csr matrix): Preprocessed features for testing.
            - y_train_val (csr matrix): True labels for training and validation.
            - y_test (csr matrix): True labels for testing.

    The function expects the following files to exist in the './data/preprocessed/' directory:
        - X_train_preprocessed.pkl
        - X_test_preprocessed.pkl
        - y_train_preprocessed.pkl
        - y_test_preprocessed.pkl

    Raises:
        FileNotFoundError: If any of the required files are missing.
    """
    print("Loading preprocessed data")
    X_train_val = joblib.load(os.path.join(DATA_DIR, "./preprocessed/X_train_preprocessed.pkl"))
    X_test = joblib.load(os.path.join(DATA_DIR, "./preprocessed/X_test_preprocessed.pkl"))

    y_train_val = joblib.load(os.path.join(DATA_DIR, "./preprocessed/y_train_preprocessed.pkl"))
    y_test = joblib.load(os.path.join(DATA_DIR, "./preprocessed/y_test_preprocessed.pkl"))
    return X_train_val, X_test, y_train_val, y_test


def data_info(df):
    """
    Displays basic information about a pandas DataFrame, including its structure and the first few rows.

    Parameters:
        df (pandas.DataFrame): The DataFrame to display information about.

    Returns:
        None
    """
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

# -----------------------------
# Load sampled data and split train/test
# -----------------------------
def load_sampled_paths_data(test_size=0.2, random_state=42):
    print("Loading data (x, y)")
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_sampled, y_train_sampled = dataset_sampler_under_oversampling(X_train, y_train)
    X_train_sampled = prepare_images_df(X_train_sampled, n_images=None, split="train", random_state=random_state)

    X_test = prepare_images_df(X_test, n_images=None, split="train", random_state=random_state)

    train_sampled_df = X_train_sampled.merge(y_train_sampled, left_index=True, right_index=True)[["processed_image_path", "image_path", "prdtypecode"]]

    test_df = X_test.merge(y_test, left_index=True, right_index=True)[["processed_image_path", "image_path", "prdtypecode"]]

    return train_sampled_df, test_df


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


def export_model(model_name, model):
    """
    Exports the given model to a file using joblib.

    Args:
        model_name (str): The name to use for the saved model file.
        model: The trained model object to be saved.

    Returns:
        None

    Side Effects:
        Saves the model to './models/{model_name}_model.pkl'.

    """
    print("Saving model")
    joblib.dump(model, f'./models/{model_name}_model.pkl')


#####################################
############# TEXT UTILS ############
#####################################

# Stopwords for text cleaning
STOP_WORDS = set(stopwords.words('english')).union(set(stopwords.words('french')))

def clean_text(text):
    """Remove HTML tags, special characters, and stopwords"""
    # Remove HTML
    text = BeautifulSoup(str(text), "html.parser").get_text()
    # Convert to lowercase
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def create_text_features(train_data, val_data, test_data, tfidf_features=10000, svd_components=300):
    """
    Create text features using TF-IDF and SVD dimensionality reduction.
    
    Args:
        train_data, val_data, test_data: DataFrames with 'text' column
        tfidf_features: Number of TF-IDF features to extract
        svd_components: Number of SVD components for dimensionality reduction
        
    Returns:
        tuple: (tfidf_vectorizer, svd_transformer, X_train_text, X_val_text, X_test_text)
    """
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=tfidf_features)
    X_train_tfidf = tfidf.fit_transform(train_data['text'])
    X_val_tfidf = tfidf.transform(val_data['text'])
    X_test_tfidf = tfidf.transform(test_data['text'])
    
    # Dimensionality reduction with SVD
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_train_text = svd.fit_transform(X_train_tfidf).astype(np.float32)
    X_val_text = svd.transform(X_val_tfidf).astype(np.float32)
    X_test_text = svd.transform(X_test_tfidf).astype(np.float32)
    
    return tfidf, svd, X_train_text, X_val_text, X_test_text


#####################################
############# PYTORCH UTILS ########
#####################################

class ProductImageDataset(Dataset):
    """Dataset for loading product images"""
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = image_path(int(row['imageid']), int(row['productid']), split="train")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = row['label_encoded']
        return image, label


class ProductMultimodalDataset(Dataset):
    """Dataset for loading product images and text features"""
    def __init__(self, dataframe, text_features, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.text_features = text_features
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = image_path(int(row['imageid']), int(row['productid']), split="train")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get text features for this sample
        text_feat = self.text_features[idx]
        
        label = row['label_encoded']
        return image, text_feat, label


def get_image_transforms():
    """Get standard image transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_resnet18_model(n_classes, freeze_layers=True):
    """Create a ResNet18 model for classification"""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    if freeze_layers:
        # Freeze all layers except layer4
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    # Replace final FC layer
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, n_classes)
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def create_multimodal_model(text_dim, num_classes, img_embed_dim=512, text_embed_dim=256, fusion_dim=256):
    """Create a multimodal fusion model (ResNet18 + Text)"""
    import torch.nn as nn
    
    class MultiModalNet(nn.Module):
        def __init__(self, text_dim, num_classes):
            super().__init__()
            
            # ========================================
            # IMAGE BRANCH (ResNet18 with fine-tuning)
            # ========================================
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            
            # Freeze early layers, unfreeze layer3 and layer4
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            
            # Remove final FC layer to get 512-dim features
            self.resnet.fc = nn.Identity()
            
            # Image projection with batch norm
            self.img_projection = nn.Sequential(
                nn.Linear(512, img_embed_dim),
                nn.BatchNorm1d(img_embed_dim),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
        
            # In-model augmentation pipeline (applied during training)
            self.train_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.90, 1.10))
            ])
            
            # ========================================
            # TEXT BRANCH (Optimized for TF-IDF+SVD features)
            # ========================================
            # TF-IDF+SVD features are already dense and informative
            # Use shallower network with higher capacity
            self.text_fc = nn.Sequential(
                nn.Linear(text_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(512, text_embed_dim),
                nn.BatchNorm1d(text_embed_dim),
            )
            
            # ========================================
            # FUSION & CLASSIFIER
            # ========================================
            combined_dim = img_embed_dim + text_embed_dim
            
            # Fusion layer with residual connection (dimensions match)
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.BatchNorm1d(combined_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            
            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.BatchNorm1d(combined_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(combined_dim // 2, combined_dim // 4),
                nn.BatchNorm1d(combined_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(combined_dim // 4, num_classes)
            )
        
        def forward(self, img, text_vec):
            # Extract features from both modalities
            img_feat = self.resnet(img)
            img_feat = self.img_projection(img_feat)
            
            text_feat = self.text_fc(text_vec)
            
            # Concatenate features
            combined = torch.cat([img_feat, text_feat], dim=1)
            
            # Fusion with residual connection
            fused = self.fusion_layer(combined) + combined
            
            # Final classification
            output = self.classifier(fused)
            return output
    
    return MultiModalNet(text_dim, num_classes)


def train_pytorch_model(model, train_loader, val_loader, device, epochs=5, learning_rate=0.0005, patience=2):
    """Train a PyTorch model with early stopping"""
    import torch.nn as nn
    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        train_loss = running_loss / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("✓ Early stopping triggered 🚦")
                break
    
    return model


def get_model_predictions(model, loader, device):
    """Get probability predictions from a PyTorch model"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    return np.vstack(all_probs)


def get_multimodal_predictions(model, loader, device):
    """Get predictions from multimodal model (image + text)"""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, text_features, _ in loader:
            images = images.to(device)
            text_features = text_features.to(device)
            outputs = model(images, text_features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_preds), np.vstack(all_probs)


def setup_image_data_loaders(train_data, val_data, test_data, batch_size=32, num_workers=2):
    """Setup data loaders for image-only training (used by late_fusion and stacking models)"""
    train_transform, val_transform = get_image_transforms()
    
    # Create datasets
    train_dataset = ProductImageDataset(train_data, train_transform)
    val_dataset = ProductImageDataset(val_data, val_transform)
    test_dataset = ProductImageDataset(test_data, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def setup_multimodal_data_loaders(train_data, val_data, test_data, X_train_text, X_val_text, X_test_text, batch_size=32, num_workers=2):
    """Setup data loaders for multimodal training (used by fusion model)"""
    train_transform, val_transform = get_image_transforms()
    
    # Create datasets
    train_dataset = ProductMultimodalDataset(train_data, X_train_text, train_transform)
    val_dataset = ProductMultimodalDataset(val_data, X_val_text, val_transform)
    test_dataset = ProductMultimodalDataset(test_data, X_test_text, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def evaluate_predictions(y_pred_encoded, y_true_encoded, label_encoder, val_data, test_data, model_name="Model"):
    """Generic function to evaluate predictions and return decoded results"""
    # Decode predictions to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Get true labels
    y_true = test_data['prdtypecode'].values
    
    # Calculate accuracy
    accuracy = (y_pred == y_true).mean()
    
    print(f"✓ {model_name} Test Accuracy: {accuracy:.4f}")
    
    return y_pred, y_true


def evaluate_multimodal_model(model, val_loader, test_loader, val_data, test_data, label_encoder, device):
    """Evaluate multimodal model and return predictions"""
    # Get predictions for validation and test sets
    y_pred_val_encoded, val_probs = get_multimodal_predictions(model, val_loader, device)
    y_pred_test_encoded, test_probs = get_multimodal_predictions(model, test_loader, device)
    
    # Decode predictions to original labels
    y_pred_val = label_encoder.inverse_transform(y_pred_val_encoded)
    y_pred_test = label_encoder.inverse_transform(y_pred_test_encoded)
    
    # Get true labels
    y_true_val = val_data['prdtypecode'].values
    y_true_test = test_data['prdtypecode'].values
    
    # Calculate accuracies
    val_accuracy = (y_pred_val == y_true_val).mean()
    test_accuracy = (y_pred_test == y_true_test).mean()
    
    print(f"✓ Validation Accuracy: {val_accuracy:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy:.4f}")
    
    return y_pred_test, y_true_test


def train_multimodal_model(model, train_loader, val_loader, device, epochs=5, learning_rate=0.0005, patience=2):
    """Train a multimodal PyTorch model with early stopping"""
    import torch.nn as nn
    import torch.optim as optim
    import copy
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, text_features, labels in train_loader:
            images = images.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, text_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        train_loss = running_loss / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, text_features, labels in val_loader:
                images = images.to(device)
                text_features = text_features.to(device)
                labels = labels.to(device)
                outputs = model(images, text_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check and best checkpoint tracking
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("✓ Early stopping triggered 🚦")
                break
    
    # Restore best model weights if available
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("✓ Restored best validation checkpoint")
    
    return model


#####################################
########## DATA PREPROCESSING #######
#####################################

def prepare_multimodal_data(tfidf_features=10000, svd_components=300, target_samples_pct=0.037):
    """
    Complete data preprocessing pipeline for multimodal models.
    
    Returns:
        tuple: (train_data, val_data, test_data, label_encoder, n_classes, 
                tfidf, svd, X_train_text, X_val_text, X_test_text)
    """
    # Load data
    X, y, _ = load_data()
    data = X.merge(y, left_index=True, right_index=True)
    print(f"✓ Total samples: {len(data)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['prdtypecode'])
    n_classes = len(label_encoder.classes_)
    print(f"✓ Number of classes: {n_classes}")
    
    # Split train/val/test (60/20/20)
    train_data, temp_data = train_test_split(
        data, test_size=0.4, random_state=42, stratify=data['label_encoded']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=temp_data['label_encoded']
    )
    print(f"✓ Train: {len(train_data)} samples")
    print(f"✓ Val: {len(val_data)} samples")
    print(f"✓ Test: {len(test_data)} samples")
    
    # Resample train set
    target_count = int(len(train_data) * target_samples_pct)
    print(f"✓ Target samples per class: {target_count}")
    
    balanced_train = []
    for class_label in train_data['label_encoded'].unique():
        class_data = train_data[train_data['label_encoded'] == class_label]
        resampled = resample(class_data, n_samples=target_count, random_state=42, replace=True)
        balanced_train.append(resampled)
    
    train_data = pd.concat(balanced_train).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✓ Train after resampling: {len(train_data)} samples")
    
    # Add boolean flag for description presence (1 if present, 0 if NA) after split/resampling
    train_data['has_description'] = train_data['description'].notna().astype(np.int8)
    val_data['has_description'] = val_data['description'].notna().astype(np.int8)
    test_data['has_description'] = test_data['description'].notna().astype(np.int8)

    # Preprocess text
    train_data['text'] = (train_data['designation'] + ' ' + train_data['description']).apply(clean_text)
    val_data['text'] = (val_data['designation'] + ' ' + val_data['description']).apply(clean_text)
    test_data['text'] = (test_data['designation'] + ' ' + test_data['description']).apply(clean_text)
    print("✓ Text preprocessing complete")
    
    # Create text features
    tfidf, svd, X_train_text, X_val_text, X_test_text = create_text_features(
        train_data, val_data, test_data, tfidf_features, svd_components
    )
    print(f"✓ TF-IDF features: {tfidf_features}")
    print(f"✓ SVD components: {svd_components}")
    print(f"✓ Final text feature shape: {X_train_text.shape}")
    
    return (train_data, val_data, test_data, label_encoder, n_classes, 
            tfidf, svd, X_train_text, X_val_text, X_test_text)
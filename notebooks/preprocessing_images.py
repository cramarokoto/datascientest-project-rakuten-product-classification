from utils import *

def prepare_first_images_df(df_X, n_images=1000, random_state=42):
    """
    Prepare a dataframe with a sample of n_images and their image paths.
    """
    X_img_sample = df_X.sample(n=n_images, random_state=random_state)
    X_img_sample["imagepath"] = X_img_sample.apply(lambda row: image_path(row["imageid"], row["productid"], "train"), axis=1)
    X_img_sample = X_img_sample["imagepath"]
    X_img_sample = pd.concat([
        X_img_sample.rename("image_path"),
        X_img_sample.apply(shape_from_path).rename("image_shape")
    ], axis=1)
    return X_img_sample
    

def ajout_content_box(df_sample):
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


def save_gray_resized_images(df_sample, resize_dim=(200, 200), folder_path=image_gray_path_train):
    """
    Save grayscale versions of images in the specified folder.
    """

    for idx, row in df_sample.iterrows():
        gray_img = Image.open(row["image_path"]).convert("L") # Convertir en niveaux de gris et Normalisation 0-1
        resized_gray_img = gray_img.resize(resize_dim) # Redimensionner l'image
        # Sauvegarder l'image
        save_path = folder_path + "/" + row["image_path"].split("/")[-1]
        resized_gray_img.save(save_path, "JPEG")
    return


def save_whole_process(df_X, n_images=1000, resize_dim=(200, 200), split="train", folder_path=image_gray_path_train, random_state=42):
    """
    Whole process to prepare images dataframe,  save grayscale resized images.
    """
    X_img_sample = prepare_images_df(df_X, n_images, split, random_state)
    save_gray_resized_images(X_img_sample, resize_dim, folder_path)
    print(n_images,"Gray resized images saved.")
    return X_img_sample


def save_preprocessed_train(resize_dim=(200, 200), folder_path=image_gray_path_train):
    save_whole_process(X_train, n_images=None, resize_dim=resize_dim, split="train",folder_path=folder_path)

def save_preprocessed_test(resize_dim=(200, 200), folder_path=image_gray_path_test):
    save_whole_process(X_test, n_images=None, resize_dim=resize_dim, split="test", folder_path=folder_path)


def normalize_image(image):
    img = np.asarray(img, dtype=np.float32) / 255.0  # Normalisation entre 0 et 1 et alège la mémoire
    return img 


def loading_preprocessed_images(n_images=1000, split="train", random_state=42):
    """
    Load preprocessed grayscale images from the specified folder.
    """
    if split == "train":
        df_X = X_train
    elif split == "test":
        df_X = X_test
    else:
        raise ValueError("split must be 'train' or 'test'")

    X_img_sample = prepare_images_df(df_X, n_images, random_state)

    images = []
    for idx, row in X_img_sample.iterrows():
        img = Image.open(row["processed_image_path"])
        img = np.asarray(img, dtype=np.float32) / 255.0  # Normalisation entre 0 et 1 et alège la mémoire
        images.append(img)

    return images

# ------------------------------------------------------------------------------------
# Pour lancer en local la sauvegarde des images prétraitées:
# - créer le dossier vide 'data/images/gray_resized_images_train' et 'data/images/gray_resized_images_test'
# - lancer les deux fonctions suivantes

# save_preprocessed_train()
# save_preprocessed_test()
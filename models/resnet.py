import time

from scripts.utils import find_image_gray_path, load_data, export_classification_reports, export_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import torchvision.io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ProductDataset(Dataset):
    def __init__(self, df, y_enc, split, transform=None):
        self.df = df.reset_index(drop=True)
        self.y = y_enc
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "imageid"]
        productid = self.df.loc[idx, "productid"]
        img_path = find_image_gray_path(img_id, productid, self.split)
        img = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.GRAY)

        if self.transform:
            img = self.transform(img)

        label = self.y[idx]
        return img, label

class RepeatChannels:
    def __call__(self, img):
        return img.repeat(3, 1, 1) # grayscale -> 3 channels

def main():
    # -----------------------------
    # 1️⃣ Load data
    # -----------------------------
    print("Loading data (x, y)")
    X, y, _ = load_data()

    df = X.merge(y, left_index=True, right_index=True)

    # -----------------------------
    # 2️⃣ Split train/val/test
    # -----------------------------
    print("Splitting data into train/val/test")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["prdtypecode"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.25, random_state=42, stratify=train_df["prdtypecode"]
    )  # 80/20/20 split

    # -----------------------------
    # 3️⃣ Encode labels
    # -----------------------------
    print("Encoding labels")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df["prdtypecode"])
    y_val_enc = le.transform(val_df["prdtypecode"])
    y_test = test_df["prdtypecode"]
    y_test_enc = le.transform(y_test)

    # dictionnaire prdtypecode -> index
    label_map = {cls: idx for idx, cls in enumerate(le.classes_)}

    # -----------------------------
    # 5️⃣ Transforms & DataLoaders
    # -----------------------------
    transformations = transforms.Compose([
        transforms.Resize((224,224)), # Size expected by the model
        RepeatChannels(), 
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    train_dataset = ProductDataset(train_df, y_train_enc, "train", transformations)
    val_dataset   = ProductDataset(val_df,   y_val_enc,   "train", transformations)
    test_dataset  = ProductDataset(test_df,  y_test_enc,  "train", transformations)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------------
    # 6️⃣ Define ResNet model
    # -----------------------------
    print("Loading pretrained ResNet18")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze backbone
    model.fc = nn.Linear(model.fc.in_features, len(le.classes_))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # -----------------------------
    # 7️⃣ Training loop
    # -----------------------------
    EPOCHS = 5
    print("Starting training")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_formatted = f"Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)"
    print(elapsed_formatted)

    # -----------------------------
    # 8️⃣ Evaluate on test set
    # -----------------------------
    print("Evaluating on test set")
    model.eval()
    y_pred_enc, y_test_enc = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            y_pred_enc.extend(predicted.cpu().numpy())
            y_test_enc.extend(labels.cpu().numpy())

    # -----------------------------
    # 9️⃣ Save model and report
    # -----------------------------
    y_pred = le.inverse_transform(y_pred_enc)

    export_model('resnet', model)
    export_classification_reports('resnet', y_pred, y_test, None, None, elapsed_formatted)

if __name__ == "__main__":
    main()
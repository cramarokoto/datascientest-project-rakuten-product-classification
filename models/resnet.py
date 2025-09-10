import os
import time
import joblib
import pandas as pd
import numpy as np

from scripts.utils import find_image_gray_path, load_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


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
)  # 60/20/20 split

# -----------------------------
# 3️⃣ Encode labels
# -----------------------------
print("Encoding labels")
le = LabelEncoder()
y_train_enc = le.fit_transform(train_df["prdtypecode"])
y_val_enc = le.transform(val_df["prdtypecode"])
y_test_enc = le.transform(test_df["prdtypecode"])

# dictionnaire prdtypecode -> index
label_map = {cls: idx for idx, cls in enumerate(le.classes_)}

# -----------------------------
# 4️⃣ Dataset class
# -----------------------------
class ProductDataset(Dataset):
    def __init__(self, df, y_enc, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.y = y_enc
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "imageid"]
        productid = self.df.loc[idx, "productid"]
        img_path = find_image_gray_path(img_id, productid, self.img_dir)
        img = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.GRAY)

        if self.transform:
            img = self.transform(img)

        label = self.y[idx]
        return img, label

# -----------------------------
# 5️⃣ Transforms & DataLoaders
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),  # grayscale -> 3 channels
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = ProductDataset(train_df, y_train_enc, "train", transform_train)
val_dataset   = ProductDataset(val_df,   y_val_enc,   "train", transform_test)
test_dataset  = ProductDataset(test_df,  y_test_enc,  "train", transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# 6️⃣ Define ResNet model
# -----------------------------
print("Loading pretrained ResNet18")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
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
print(f"Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/60:.2f} minutes)")

# -----------------------------
# 8️⃣ Evaluate on test set
# -----------------------------
print("Evaluating on test set")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=[str(c) for c in le.classes_], output_dict=True)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# -----------------------------
# 9️⃣ Save model and report
# -----------------------------
print("Saving model and classification report")
os.makedirs("./models", exist_ok=True)
torch.save(model.state_dict(), "./models/resnet18_model.pth")
joblib.dump(le, "./models/label_encoder.pkl")

with open("./models/resnet18_classification_report.txt", "w") as f:
    f.write(classification_report(all_labels, all_preds, target_names=[str(c) for c in le.classes_]))

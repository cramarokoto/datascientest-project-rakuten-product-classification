import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import joblib

from scripts.utils import load_data, export_classification_reports, image_path

# ========================================
# CONFIGURATION
# ========================================
MODEL_SAVE_PATH = 'models/late_fusion_model_text.pkl'
RESNET_SAVE_PATH = 'models/late_fusion_model_image.pth'

# Stopwords for text cleaning
STOP_WORDS = set(stopwords.words('english')).union(set(stopwords.words('french')))

# ========================================
# DATASET CLASS
# ========================================
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

# ========================================
# TEXT PREPROCESSING
# ========================================
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

# ========================================
# MAIN PIPELINE
# ========================================
def main():
    start_time = time.time()
    
    # -----------------------------
    # 1️⃣ Load data
    # -----------------------------
    print("-" * 80)
    print("Loading data (X, y)")
    print("-" * 80)
    
    X, y, _ = load_data()
    data = X.merge(y, left_index=True, right_index=True)
    print(f"✓ Total samples: {len(data)}")
    
    # -----------------------------
    # 2️⃣ Encode labels
    # -----------------------------
    print("-" * 80)
    print("Encoding labels")
    print("-" * 80)
    
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['prdtypecode'])
    n_classes = len(label_encoder.classes_)
    print(f"✓ Number of classes: {n_classes}")
    print(f"✓ Classes: {list(label_encoder.classes_)}")
    
    # -----------------------------
    # 3️⃣ Split train/val/test (60/20/20)
    # -----------------------------
    print("-" * 80)
    print("Splitting data into train/val/test (60/20/20)")
    print("-" * 80)
    
    train_data, temp_data = train_test_split(
        data, test_size=0.4, random_state=42, stratify=data['label_encoded']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=temp_data['label_encoded']
    )
    print(f"✓ Train: {len(train_data)} samples")
    print(f"✓ Val: {len(val_data)} samples")
    print(f"✓ Test: {len(test_data)} samples")
    
    # -----------------------------
    # 4️⃣ Resample train set (3.7% per class)
    # -----------------------------
    print("-" * 80)
    print("Resampling train set to balance classes (3.7% per class)")
    print("-" * 80)
    
    target_count = int(len(train_data) * 0.037)
    print(f"✓ Target samples per class: {target_count}")
    
    balanced_train = []
    for class_label in train_data['label_encoded'].unique():
        class_data = train_data[train_data['label_encoded'] == class_label]
        resampled = resample(class_data, n_samples=target_count, random_state=42, replace=True)
        balanced_train.append(resampled)
    
    train_data = pd.concat(balanced_train).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✓ Train after resampling: {len(train_data)} samples")
    
    # -----------------------------
    # 5️⃣ Preprocess text data
    # -----------------------------
    print("-" * 80)
    print("Preprocessing text (remove HTML & stopwords)")
    print("-" * 80)
    
    # Combine designation and description
    train_data['text'] = (train_data['designation'] + ' ' + train_data['description'].fillna('')).apply(clean_text)
    val_data['text'] = (val_data['designation'] + ' ' + val_data['description'].fillna('')).apply(clean_text)
    test_data['text'] = (test_data['designation'] + ' ' + test_data['description'].fillna('')).apply(clean_text)
    print("✓ Text preprocessing complete")
    
    # -----------------------------
    # 6️⃣ Create text features (TF-IDF + SVD)
    # -----------------------------
    print("-" * 80)
    print("Creating text features (TF-IDF + SVD)")
    print("-" * 80)
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf.fit_transform(train_data['text'])
    X_val_tfidf = tfidf.transform(val_data['text'])
    X_test_tfidf = tfidf.transform(test_data['text'])
    
    # Dimensionality reduction with SVD
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train_text = svd.fit_transform(X_train_tfidf)
    X_val_text = svd.transform(X_val_tfidf)
    X_test_text = svd.transform(X_test_tfidf)
    
    print(f"✓ TF-IDF features: 10000")
    print(f"✓ SVD components: 300")
    print(f"✓ Final text feature shape: {X_train_text.shape}")
    
    # -----------------------------
    # 7️⃣ Train XGBoost on text features
    # -----------------------------
    print("-" * 80)
    print("Training XGBoost on text features")
    print("-" * 80)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_text, train_data['label_encoded'])
    
    train_acc_text = xgb_model.score(X_train_text, train_data['label_encoded'])
    val_acc_text = xgb_model.score(X_val_text, val_data['label_encoded'])
    
    print(f"✓ XGBoost Train Accuracy: {train_acc_text:.4f}")
    print(f"✓ XGBoost Val Accuracy: {val_acc_text:.4f}")
    
    # Get text model predictions (probabilities)
    text_probs_val = xgb_model.predict_proba(X_val_text)
    text_probs_test = xgb_model.predict_proba(X_test_text)
    
    # -----------------------------
    # 8️⃣ Prepare image datasets
    # -----------------------------
    print("-" * 80)
    print("Preparing image datasets")
    print("-" * 80)
    
    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ProductImageDataset(train_data, train_transform)
    val_dataset = ProductImageDataset(val_data, val_transform)
    test_dataset = ProductImageDataset(test_data, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # -----------------------------
    # 9️⃣ Define ResNet18 model
    # -----------------------------
    print("-" * 80)
    print("Loading pretrained ResNet18 for fine-tuning")
    print("-" * 80)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load pretrained ResNet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze all layers except layer4
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    print(f"✓ Model loaded with {n_classes} output classes")
    print("✓ Layer4 + FC unfrozen for training")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # -----------------------------
    # 🔟 Train ResNet18
    # -----------------------------
    print("-" * 80)
    print("Training ResNet18 on images")
    print("-" * 80)
    
    EPOCHS = 5
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 2
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
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
    
    # -----------------------------
    # 1️⃣1️⃣ Get image model predictions
    # -----------------------------
    print("-" * 80)
    print("Getting image model predictions")
    print("-" * 80)
    
    def get_image_probabilities(loader):
        """Get probability predictions from image model"""
        model.eval()
        all_probs = []
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    image_probs_val = get_image_probabilities(val_loader)
    image_probs_test = get_image_probabilities(test_loader)
    print("✓ Image predictions computed")
    
    # -----------------------------
    # 1️⃣2️⃣ Late fusion (combine predictions)
    # -----------------------------
    print("-" * 80)
    print("Late fusion (averaging text + image probabilities)")
    print("-" * 80)

    # Average probabilities from text and image models
    final_probs_val = (text_probs_val + image_probs_val) / 2
    final_probs_test = (text_probs_test + image_probs_test) / 2
    
    # Get final predictions
    y_pred_val_encoded = final_probs_val.argmax(axis=1)
    y_pred_test_encoded = final_probs_test.argmax(axis=1)
    
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
    
    # -----------------------------
    # 1️⃣3️⃣ Save models
    # -----------------------------
    print("-" * 80)
    print("Saving models")
    print("-" * 80)
    
    # Save text model components
    text_models = {
        'xgb': xgb_model,
        'tfidf': tfidf,
        'svd': svd,
        'label_encoder': label_encoder
    }
    joblib.dump(text_models, MODEL_SAVE_PATH)
    
    # Save image model
    torch.save(model.state_dict(), RESNET_SAVE_PATH)
    
    print(f"✓ Text model saved to {MODEL_SAVE_PATH}")
    print(f"✓ Image model saved to {RESNET_SAVE_PATH}")
    
    # -----------------------------
    # 1️⃣4️⃣ Export classification report
    # -----------------------------
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_formatted = f"{elapsed:.2f} secondes ({elapsed/60:.2f} minutes)"
    
    print("-" * 80)
    print("Exporting classification report")
    print("-" * 80)
    
    print(f"✓ Total execution time: {elapsed_formatted}")
    
    # Generate and save classification report
    export_classification_reports('late_fusion', y_pred_test, y_true_test, None, None, elapsed_formatted)
    
    print("✅ TRAINING COMPLETE!")
    


if __name__ == "__main__":
    main()
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import torch
import joblib

from scripts.utils import (
    prepare_multimodal_data, export_classification_reports, 
    create_resnet18_model, train_pytorch_model, get_model_predictions,
    setup_image_data_loaders, evaluate_predictions
)

# ========================================
# CONFIGURATION
# ========================================
TEXT_MODEL_PATH = 'models/stacking_model_text.pkl'
IMAGE_MODEL_PATH = 'models/stacking_model_image.pth'
META_MODEL_PATH = 'models/stacking_model_meta.pkl'

# Training parameters
TFIDF_FEATURES = 10000
SVD_COMPONENTS = 300
TARGET_SAMPLES_PCT = 0.037
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
PATIENCE = 2


# ========================================
# MAIN PIPELINE
# ========================================
def main():
    start_time = time.time()
    
    # -----------------------------
    # 1. Prepare multimodal data
    # -----------------------------
    print("=" * 80)
    print("1. Preparing multimodal data")
    print("=" * 80)
    
    (train_data, val_data, test_data, label_encoder, n_classes, 
     tfidf, svd, X_train_text, X_val_text, X_test_text) = prepare_multimodal_data(
        tfidf_features=TFIDF_FEATURES,
        svd_components=SVD_COMPONENTS,
        target_samples_pct=TARGET_SAMPLES_PCT
    )
    
    # -----------------------------
    # 2. Train XGBoost on text features
    # -----------------------------
    print("\n" + "=" * 80)
    print("2. Training XGBoost on text features")
    print("=" * 80)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.2,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_text, train_data['label_encoded'])
    
    train_acc_text = xgb_model.score(X_train_text, train_data['label_encoded'])
    val_acc_text = xgb_model.score(X_val_text, val_data['label_encoded'])
    
    print(f"XGBoost Train Accuracy: {train_acc_text:.4f}")
    print(f"XGBoost Val Accuracy: {val_acc_text:.4f}")
    
    # Get text model predictions (probabilities)
    text_probs_train = xgb_model.predict_proba(X_train_text)
    text_probs_val = xgb_model.predict_proba(X_val_text)
    text_probs_test = xgb_model.predict_proba(X_test_text)
    
    # -----------------------------
    # 3. Prepare image datasets
    # -----------------------------
    print("\n" + "=" * 80)
    print("3. Preparing image datasets")
    print("=" * 80)
    
    train_loader, val_loader, test_loader = setup_image_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    
    # -----------------------------
    # 4. Define ResNet18 model
    # -----------------------------
    print("\n" + "=" * 80)
    print("4. Loading pretrained ResNet18 for fine-tuning")
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create ResNet18 model
    model = create_resnet18_model(n_classes, freeze_layers=True)
    model = model.to(device)
    
    print(f"Model loaded with {n_classes} output classes")
    print("Layer4 + FC unfrozen for training")
    
    # -----------------------------
    # 5. Train ResNet18
    # -----------------------------
    print("\n" + "=" * 80)
    print("5. Training ResNet18 on images")
    print("=" * 80)
    
    # Train the model using the utility function
    model = train_pytorch_model(
        model, train_loader, val_loader, device, 
        epochs=EPOCHS, learning_rate=LEARNING_RATE, patience=PATIENCE
    )
    
    # -----------------------------
    # 6. Get image model predictions
    # -----------------------------
    print("\n" + "=" * 80)
    print("6. Getting image model predictions")
    print("=" * 80)
    
    # Get image model predictions using utility function
    image_probs_train = get_model_predictions(model, train_loader, device)
    image_probs_val = get_model_predictions(model, val_loader, device)
    image_probs_test = get_model_predictions(model, test_loader, device)
    print("Image predictions computed")
    
    # -----------------------------
    # 7. Stacking - Train meta-classifier
    # -----------------------------
    print("\n" + "=" * 80)
    print("7. Stacking - Training meta-classifier")
    print("=" * 80)
    
    # Stack predictions as features (concatenate text + image probabilities)
    X_stack_train = np.hstack([text_probs_train, image_probs_train])
    X_stack_val = np.hstack([text_probs_val, image_probs_val])
    X_stack_test = np.hstack([text_probs_test, image_probs_test])
    
    print(f"Stacked feature shape: {X_stack_train.shape}")
    print(f"Features: {n_classes} text probs + {n_classes} image probs = {n_classes * 2} total")
    
    # Train meta-classifier (Logistic Regression)
    meta_model = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    meta_model.fit(X_stack_train, train_data['label_encoded'])
    
    # Evaluate meta-classifier
    train_acc_meta = meta_model.score(X_stack_train, train_data['label_encoded'])
    val_acc_meta = meta_model.score(X_stack_val, val_data['label_encoded'])
    
    print(f"Meta-classifier Train Accuracy: {train_acc_meta:.4f}")
    print(f"Meta-classifier Val Accuracy: {val_acc_meta:.4f}")
    
    # Get final predictions
    y_pred_val_encoded = meta_model.predict(X_stack_val)
    y_pred_test_encoded = meta_model.predict(X_stack_test)
    
    # Evaluate predictions using utility function
    y_pred_test, y_true_test = evaluate_predictions(
        y_pred_test_encoded, None, label_encoder, val_data, test_data, "Stacking"
    )
    
    # -----------------------------
    # 8. Save models
    # -----------------------------
    print("\n" + "=" * 80)
    print("8. Saving models")
    print("=" * 80)
    
    # Save text model components
    text_models = {
        'xgb': xgb_model,
        'tfidf': tfidf,
        'svd': svd,
        'label_encoder': label_encoder
    }
    joblib.dump(text_models, TEXT_MODEL_PATH)
    
    # Save image model
    torch.save(model.state_dict(), IMAGE_MODEL_PATH)
    
    # Save meta-classifier
    joblib.dump(meta_model, META_MODEL_PATH)
    
    print(f"Text model saved to {TEXT_MODEL_PATH}")
    print(f"Image model saved to {IMAGE_MODEL_PATH}")
    print(f"Meta-classifier saved to {META_MODEL_PATH}")
    
    # -----------------------------
    # 9. Export classification report
    # -----------------------------
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_formatted = f"{elapsed:.2f} secondes ({elapsed/60:.2f} minutes)"
    
    print("\n" + "=" * 80)
    print("9. Exporting classification report")
    print("=" * 80)
    print(f"Total execution time: {elapsed_formatted}")
    
    # Export classification report
    export_classification_reports('stacking', y_pred_test, y_true_test, None, None, elapsed_formatted)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
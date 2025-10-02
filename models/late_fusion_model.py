import time
import numpy as np
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
MODEL_SAVE_PATH = 'models/late_fusion_model_text.pkl'
RESNET_SAVE_PATH = 'models/late_fusion_model_image.pth'

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
    # 1️⃣ Prepare multimodal data
    # -----------------------------
    print("-" * 80)
    print("Preparing multimodal data")
    print("-" * 80)
    
    (train_data, val_data, test_data, label_encoder, n_classes, 
     tfidf, svd, X_train_text, X_val_text, X_test_text) = prepare_multimodal_data(
        tfidf_features=TFIDF_FEATURES,
        svd_components=SVD_COMPONENTS,
        target_samples_pct=TARGET_SAMPLES_PCT
    )
    
    # -----------------------------
    # 2️⃣ Train XGBoost on text features
    # -----------------------------
    print("-" * 80)
    print("Training XGBoost on text features")
    print("-" * 80)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
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
    # 3️⃣ Prepare image datasets
    # -----------------------------
    print("-" * 80)
    print("Preparing image datasets")
    print("-" * 80)
    
    train_loader, val_loader, test_loader = setup_image_data_loaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    
    # -----------------------------
    # 4️⃣ Define ResNet18 model
    # -----------------------------
    print("-" * 80)
    print("Loading pretrained ResNet18 for fine-tuning")
    print("-" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Create ResNet18 model
    model = create_resnet18_model(n_classes, freeze_layers=True)
    model = model.to(device)
    
    print(f"✓ Model loaded with {n_classes} output classes")
    print("✓ Layer4 + FC unfrozen for training")
    
    # -----------------------------
    # 5️⃣ Train ResNet18
    # -----------------------------
    print("-" * 80)
    print("Training ResNet18 on images")
    print("-" * 80)
    
    # Train the model using the utility function
    model = train_pytorch_model(
        model, train_loader, val_loader, device, 
        epochs=EPOCHS, learning_rate=LEARNING_RATE, patience=PATIENCE
    )
    
    # -----------------------------
    # 6️⃣ Get image model predictions
    # -----------------------------
    print("-" * 80)
    print("Getting image model predictions")
    print("-" * 80)
    
    # Get image model predictions using utility function
    image_probs_val = get_model_predictions(model, val_loader, device)
    image_probs_test = get_model_predictions(model, test_loader, device)
    print("✓ Image predictions computed")
    
    # -----------------------------
    # 7️⃣ Late fusion (combine predictions)
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
    
    # Evaluate predictions using utility function
    y_pred_test, y_true_test = evaluate_predictions(
        y_pred_test_encoded, None, label_encoder, val_data, test_data, "Late Fusion"
    )
    
    # -----------------------------
    # 8️⃣ Save models
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
    # 9️⃣ Export classification report
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
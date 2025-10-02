import time
import torch
import joblib
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from scripts.utils import (
    prepare_multimodal_data, export_classification_reports, 
    create_multimodal_model, train_multimodal_model, 
    setup_multimodal_data_loaders, evaluate_multimodal_model
)

# ========================================
# CONFIGURATION
# ========================================
class FusionConfig:
    # Model paths
    MODEL_SAVE_PATH = 'models/fusion_model.pth'
    TEXT_MODEL_SAVE_PATH = 'models/fusion_model_text.pkl'
    
    # Dimensions
    TFIDF_FEATURES = 10000
    SVD_COMPONENTS = 300
    TEXT_EMBED_DIM = 256
    IMG_EMBED_DIM = 512
    FUSION_DIM = 256
    
    # Training parameters
    TARGET_SAMPLES_PCT = 0.037
    EPOCHS = 15
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0007
    PATIENCE = 4
    NUM_WORKERS = 15

# ========================================
# MAIN PIPELINE
# ========================================
def main():
    start_time = time.time()
    config = FusionConfig()
    
    # -----------------------------
    # 1️⃣ Prepare multimodal data
    # -----------------------------
    print("-" * 80)
    print("Preparing multimodal data")
    print("-" * 80)
    
    (train_data, val_data, test_data, label_encoder, n_classes, 
     tfidf, svd, X_train_text, X_val_text, X_test_text) = prepare_multimodal_data(
        tfidf_features=config.TFIDF_FEATURES,
        svd_components=config.SVD_COMPONENTS,
        target_samples_pct=config.TARGET_SAMPLES_PCT
    )
    
    # Append has_description as an additional numeric text feature
    train_desc = train_data['has_description'].astype(np.float32).to_numpy().reshape(-1, 1)
    val_desc = val_data['has_description'].astype(np.float32).to_numpy().reshape(-1, 1)
    test_desc = test_data['has_description'].astype(np.float32).to_numpy().reshape(-1, 1)

    X_train_text = np.hstack([X_train_text, train_desc]).astype(np.float32)
    X_val_text = np.hstack([X_val_text, val_desc]).astype(np.float32)
    X_test_text = np.hstack([X_test_text, test_desc]).astype(np.float32)
    
    # -----------------------------
    # 2️⃣ Setup data loaders
    # -----------------------------
    print("-" * 80)
    print("Preparing multimodal datasets")
    print("-" * 80)
    
    train_loader, val_loader, test_loader = setup_multimodal_data_loaders(
        train_data, val_data, test_data, X_train_text, X_val_text, X_test_text, 
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )
    
    # -----------------------------
    # 3️⃣ Create and train model
    # -----------------------------
    print("-" * 80)
    print("Loading multimodal fusion model (ResNet18 + Text)")
    print("-" * 80)
    
    # Detect available device (CUDA, ROCm, or CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Works for both NVIDIA CUDA and AMD ROCm
        print(f"✓ Device: {device}")
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("⚠ No GPU detected, using CPU")
    
    # Create multimodal model
    model = create_multimodal_model(
        text_dim=X_train_text.shape[1], 
        num_classes=n_classes,
        img_embed_dim=config.IMG_EMBED_DIM,
        text_embed_dim=config.TEXT_EMBED_DIM,
        fusion_dim=config.FUSION_DIM
    )
    model = model.to(device)
    
    # Enable multi-GPU training if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"✓ Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    
    print(f"✓ Model loaded with {n_classes} output classes")
    print("✓ ResNet18 + Text fusion architecture")
    
    # Train model
    print("-" * 80)
    print("Training multimodal fusion model")
    print("-" * 80)
    
    model = train_multimodal_model(
        model, train_loader, val_loader, device,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.PATIENCE
    )
    
    # -----------------------------
    # 4️⃣ Evaluate and save
    # -----------------------------
    print("-" * 80)
    print("Getting final predictions")
    print("-" * 80)
    
    y_pred_test, y_true_test = evaluate_multimodal_model(
        model, val_loader, test_loader, val_data, test_data, label_encoder, device
    )
    
    # Save models
    print("-" * 80)
    print("Saving models")
    print("-" * 80)
    
    # Save underlying module when using DataParallel
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(model_state, config.MODEL_SAVE_PATH)
    
    text_models = {
        'tfidf': tfidf,
        'svd': svd,
        'label_encoder': label_encoder
    }
    joblib.dump(text_models, config.TEXT_MODEL_SAVE_PATH)
    
    print(f"✓ Multimodal model saved to {config.MODEL_SAVE_PATH}")
    print(f"✓ Text preprocessing saved to {config.TEXT_MODEL_SAVE_PATH}")
    
    # Export results
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_formatted = f"{elapsed:.2f} secondes ({elapsed/60:.2f} minutes)"
    
    print("-" * 80)
    print("Exporting classification report")
    print("-" * 80)
    
    print(f"✓ Total execution time: {elapsed_formatted}")
    export_classification_reports('fusion', y_pred_test, y_true_test, None, None, elapsed_formatted)
    
    print("✅ TRAINING COMPLETE!")


if __name__ == "__main__":
    main()

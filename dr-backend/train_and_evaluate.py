import os
import cv2
import pandas as pd
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

def extract_features(image_path):
    """
    Extracts GLCM, LBP, and Color features matching the pre-processing pipeline.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    # Convert BGR (OpenCV default) to RGB
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (256, 256))

    # --- BRANCH 1: GREEN CHANNEL & TEXTURE FEATURES ---
    green_channel = rgb_img[:, :, 1]

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(green_channel)

    # A. GLCM Features (0°, 45°, 90°, 135°)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(enhanced_g, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)

    glcm_contrast = graycoprops(glcm, 'contrast').mean()
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    glcm_homogeneity = graycoprops(glcm, 'homogeneity').mean()
    glcm_energy = graycoprops(glcm, 'energy').mean()
    glcm_correlation = graycoprops(glcm, 'correlation').mean()

    glcm_features = [glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation]

    # B. LBP Features
    P, R = 8, 1
    lbp = local_binary_pattern(enhanced_g, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # --- BRANCH 2: COLOR FEATURES (Original RGB Image) ---
    color_features = []
    for channel_idx in range(3):  # R, G, B
        ch_data = rgb_img[:, :, channel_idx]
        color_features.extend([
            float(np.mean(ch_data)),
            float(np.var(ch_data)),
            float(np.std(ch_data)),
            float(np.min(ch_data)),
            float(np.max(ch_data))
        ])

    # --- COMBINED FEATURE VECTOR ---
    combined_vector = np.hstack([glcm_features, lbp_hist, color_features])
    return combined_vector

def load_dataset_features(csv_path, image_dir):
    """Loads a CSV and extracts features for all valid images in the directory."""
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return np.array([]), np.array([])
        
    df = pd.read_csv(csv_path)
    image_col = df.columns[0]
    label_col = df.columns[1]
    
    X, y = [], []
    for _, row in df.iterrows():
        img_name = str(row[image_col]).strip()
        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'
            
        img_path = os.path.join(image_dir, img_name)
        feats = extract_features(img_path)
        
        if feats is not None:
            X.append(feats)
            y.append(row[label_col])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    TRAIN_CSV = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/diabetic-retinopathy/dr-backend/datasets/train_1.csv"
    TRAIN_DIR = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/diabetic-retinopathy/dr-backend/datasets/train_images/train_images"
    
    VAL_CSV = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/diabetic-retinopathy/dr-backend/datasets/valid.csv"
    VAL_DIR = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/diabetic-retinopathy/dr-backend/datasets/val_images/val_images"
    
    TEST_CSV = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/diabetic-retinopathy/dr-backend/datasets/test.csv"
    TEST_DIR = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/diabetic-retinopathy/dr-backend/datasets/test_images/test_images"

    print("--- STEP 1: Processing Training Data ---")
    X_train, y_train = load_dataset_features(TRAIN_CSV, TRAIN_DIR)
    print(f"Loaded {len(X_train)} training samples with feature vector length: {X_train.shape[1] if len(X_train) > 0 else 0}")

    print("\n--- STEP 2: Processing Validation Data ---")
    X_val, y_val = load_dataset_features(VAL_CSV, VAL_DIR)
    print(f"Loaded {len(X_val)} validation samples.")

    # Combine Train + Validation sets
    if len(X_val) > 0:
        X_full = np.vstack((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))
    else:
        print("Validation set empty or missing CSV. Training using train set only.")
        X_full, y_full = X_train, y_train

    print(f"\nTotal combined training dataset size: {len(X_full)} images.")

    print("\n--- STEP 3: Training Final Random Forest Model ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_full, y_full)

    # Overwrite rf_model.pkl so FastAPI immediately serves the best model
    joblib.dump(rf_model, "rf_model.pkl")
    print("Saved updated model to 'rf_model.pkl'.")

    print("\n--- STEP 4: Evaluating on Test Set ---")
    X_test, y_test = load_dataset_features(TEST_CSV, TEST_DIR)
    
    if len(X_test) > 0:
        y_pred = rf_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
        
        print("\n" + "=" * 45)
        print("           FINAL TEST RESULTS           ")
        print("=" * 45)
        print(f"Accuracy:                  {acc * 100:.2f}%")
        print(f"Quadratic Weighted Kappa:  {qwk:.4f}")
        print("=" * 45)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"]))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    else:
        print("Note: Test dataset unlabelled or missing.")
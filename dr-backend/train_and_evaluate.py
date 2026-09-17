import os
import cv2
import pandas as pd
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score

def extract_features(image_path):
    """Extracts GLCM texture features and color statistics from a fundus image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    
    # Texture Features (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # Intensity Statistics
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    return [contrast, homogeneity, energy, mean_val, std_val]

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
    # Define file paths for train, validation, and test sets
    # Update CSV names if yours differ (e.g., val.csv, test.csv)
    TRAIN_CSV = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/dr-backend/datasets/train_1.csv"
    TRAIN_DIR = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/dr-backend/datasets/train_images/train_images"
    
    VAL_CSV = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/dr-backend/datasets/valid.csv"
    VAL_DIR = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/dr-backend/datasets/val_images/val_images"
    
    TEST_CSV = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/dr-backend/datasets/test.csv"
    TEST_DIR = "C:/Users/SAHIL SINGH/OneDrive/Desktop/Mini Project/dr-backend/datasets/test_images/test_images"

    print("--- STEP 1: Processing Training Data ---")
    X_train, y_train = load_dataset_features(TRAIN_CSV, TRAIN_DIR)
    print(f"Loaded {len(X_train)} training samples.")

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
    print("Saved final model to 'rf_model.pkl'.")

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
        print("Note: If test.csv is unlabelled, run predictions on test images to generate your final submission file.")
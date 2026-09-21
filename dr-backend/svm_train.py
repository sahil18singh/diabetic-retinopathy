import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Point DATA_DIR directly to the aptos folder
DATA_DIR = os.path.join(".", "aptos")

TRAIN_CSV = os.path.join(DATA_DIR, "train_1.csv")
VAL_CSV = os.path.join(DATA_DIR, "valid.csv")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val_images")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================
# 2. FEATURE EXTRACTOR SETUP (ResNet18)
# ==========================================
# We use ResNet18 trained on ImageNet to convert images into 512-dimensional feature vectors.
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Remove the final classification layer (fc)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

# Image Transformations matching PyTorch defaults
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 3. HELPER FUNCTION TO EXTRACT FEATURES
# ==========================================
def extract_features(df, img_dir):
    features = []
    labels = []
    
    # 1. Identify correct ID column
    id_col = 'id_code' if 'id_code' in df.columns else df.columns[0]
    # 2. Identify correct label column
    label_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[1]

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting from {os.path.basename(img_dir)}"):
            img_id = str(row[id_col]).strip()
            
            # Find matching file regardless of extension
            img_path = None
            if os.path.exists(os.path.join(img_dir, img_id)):
                img_path = os.path.join(img_dir, img_id)
            elif os.path.exists(os.path.join(img_dir, f"{img_id}.png")):
                img_path = os.path.join(img_dir, f"{img_id}.png")
            elif os.path.exists(os.path.join(img_dir, f"{img_id}.jpg")):
                img_path = os.path.join(img_dir, f"{img_id}.jpg")
            elif os.path.exists(os.path.join(img_dir, f"{img_id}.jpeg")):
                img_path = os.path.join(img_dir, f"{img_id}.jpeg")

            if img_path is None:
                continue

            # Load image and extract features
            image = Image.open(img_path).convert('RGB')
            tensor_img = transform(image).unsqueeze(0).to(DEVICE)
            
            feat = feature_extractor(tensor_img)
            feat = torch.flatten(feat, 1).cpu().numpy().squeeze()
            
            features.append(feat)
            labels.append(row[label_col])
            
    # Raise clear error if no images were matched
    if len(features) == 0:
        raise ValueError(f"No images were found in directory '{img_dir}'. "
                         f"Please verify folder paths and file extensions.")

    return np.array(features), np.array(labels)

# ==========================================
# 4. LOAD DATA & EXTRACT FEATURES
# ==========================================
train_df = pd.read_csv(TRAIN_CSV)
print("CSV Columns:", train_df.columns.tolist())
print("First few image IDs in CSV:", train_df.iloc[:, 0].head().tolist())

# Inspect actual folder contents
if os.path.exists(TRAIN_IMG_DIR):
    print("Sample actual files in folder:", os.listdir(TRAIN_IMG_DIR)[:5])
else:
    print(f"Directory not found: {TRAIN_IMG_DIR}")


val_df = pd.read_csv(VAL_CSV)

print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

X_train, y_train = extract_features(train_df, TRAIN_IMG_DIR)
X_val, y_val = extract_features(val_df, VAL_IMG_DIR)

# Normalize/Standardize extracted features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==========================================
# 5. TRAIN SVM MODEL
# ==========================================
print("\nTraining SVM Classifier...")
# Support Vector Classifier with RBF kernel and balanced class weights
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# ==========================================
# 6. EVALUATION
# ==========================================
y_pred = svm_classifier.predict(X_val_scaled)

print("\n--- Validation Results ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%\n")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=[
    "0 - No DR", 
    "1 - Mild", 
    "2 - Moderate", 
    "3 - Severe", 
    "4 - Proliferative"
]))
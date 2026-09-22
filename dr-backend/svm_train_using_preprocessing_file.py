import os
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

from app.preprocess import extract_ml_features

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================

DATA_DIR = os.path.join(".", "aptos")

TRAIN_CSV = os.path.join(
    DATA_DIR,
    "train_1.csv"
)

VAL_CSV = os.path.join(
    DATA_DIR,
    "valid.csv"
)

TRAIN_IMG_DIR = os.path.join(
    DATA_DIR,
    "train_images"
)

VAL_IMG_DIR = os.path.join(
    DATA_DIR,
    "val_images"
)


# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================

def extract_features(df, img_dir):

    features = []
    labels = []

    # Identify columns
    id_col = (
        'id_code'
        if 'id_code' in df.columns
        else df.columns[0]
    )

    label_col = (
        'diagnosis'
        if 'diagnosis' in df.columns
        else df.columns[1]
    )

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Extracting from {os.path.basename(img_dir)}"
    ):

        img_id = str(
            row[id_col]
        ).strip()

        # --------------------------------------
        # Find image
        # --------------------------------------

        img_path = None

        possible_extensions = [
            "",
            ".png",
            ".jpg",
            ".jpeg"
        ]

        for ext in possible_extensions:

            candidate = os.path.join(
                img_dir,
                img_id + ext
            )

            if os.path.exists(candidate):

                img_path = candidate
                break

        if img_path is None:
            continue

        # --------------------------------------
        # Load image
        # --------------------------------------

        image = Image.open(
            img_path
        ).convert("RGB")

        # --------------------------------------
        # Extract handcrafted features
        # --------------------------------------

        feature_vector = extract_ml_features(
            image
        )

        # extract_ml_features returns (1, N)
        feature_vector = feature_vector.squeeze()

        features.append(
            feature_vector
        )

        labels.append(
            row[label_col]
        )

    # --------------------------------------
    # Check
    # --------------------------------------

    if len(features) == 0:

        raise ValueError(
            f"No images were found in "
            f"'{img_dir}'"
        )

    return (
        np.array(features),
        np.array(labels)
    )


# ==========================================
# 3. LOAD DATA
# ==========================================

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

print(
    f"Loaded {len(train_df)} training samples "
    f"and {len(val_df)} validation samples."
)


# ==========================================
# 4. EXTRACT FEATURES
# ==========================================

print("\nExtracting training features...")

X_train, y_train = extract_features(
    train_df,
    TRAIN_IMG_DIR
)

print("\nExtracting validation features...")

X_val, y_val = extract_features(
    val_df,
    VAL_IMG_DIR
)


print("\nFeature shapes:")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)


# ==========================================
# 5. STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)


# ==========================================
# 6. TRAIN SVM
# ==========================================

print("\nTraining SVM Classifier...")

svm_classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    random_state=42
)

svm_classifier.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = svm_classifier.predict(
    X_val_scaled
)


# ==========================================
# 8. EVALUATION
# ==========================================

print("\n--- Validation Results ---")

print(
    f"Accuracy: "
    f"{accuracy_score(y_val, y_pred) * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        y_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "0 - No DR",
            "1 - Mild",
            "2 - Moderate",
            "3 - Severe",
            "4 - Proliferative"
        ]
    )
)
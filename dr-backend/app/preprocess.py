import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_ml_features(image: Image.Image) -> np.ndarray:
    """
    Extracts GLCM, LBP, and Color features following the flowchart pipeline.
    Expects a PIL Image in RGB format.
    """
    # 1. Convert PIL image to NumPy RGB array & resize
    rgb_img = np.array(image.convert("RGB"))
    rgb_img = cv2.resize(rgb_img, (256, 256))

    # --- BRANCH 1: GREEN CHANNEL & TEXTURE FEATURES ---
    # Extract Green Channel (Index 1 in RGB)
    green_channel = rgb_img[:, :, 1]

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(green_channel)

    # A. GLCM Texture Features (0°, 45°, 90°, 135°)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(enhanced_g, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)

    glcm_contrast = graycoprops(glcm, 'contrast').mean()
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    glcm_homogeneity = graycoprops(glcm, 'homogeneity').mean()
    glcm_energy = graycoprops(glcm, 'energy').mean()
    glcm_correlation = graycoprops(glcm, 'correlation').mean()

    glcm_features = [glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation]

    # B. LBP Texture Features (Uniform Patterns)
    P, R = 8, 1
    lbp = local_binary_pattern(enhanced_g, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # --- BRANCH 2: COLOR FEATURES (Original RGB Image) ---
    color_features = []
    for channel_idx in range(3):  # Red, Green, Blue
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

    # Return shape (1, N) for scikit-learn model input
    return np.array([combined_vector])
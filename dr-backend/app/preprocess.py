import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

def extract_ml_features(image: Image.Image):
    cv_img = np.array(image)
    
    # Grayscale check
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img
        
    gray = cv2.resize(gray, (256, 256))
    
    # GLCM Texture Features
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Return as 2D array for scikit-learn: [[f1, f2, f3, f4, f5]]
    return np.array([[contrast, homogeneity, energy, mean_val, std_val]])
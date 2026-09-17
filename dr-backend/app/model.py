import os
import joblib

DR_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def load_ml_model(weights_path: str = "rf_model.pkl"):
    if os.path.exists(weights_path):
        return joblib.load(weights_path)
    print("Warning: rf_model.pkl not found! Run train_ml.py first.")
    return None

def run_ml_inference(model, feature_array):
    if model is None:
        return "Model not found", 0.0
        
    pred_idx = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    
    return DR_CLASSES[pred_idx], float(probabilities[pred_idx])
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.preprocess import extract_ml_features
from app.model import load_ml_model, run_ml_inference

app = FastAPI(title="Diabetic Retinopathy Diagnostic System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Random Forest model at startup
ml_model = load_ml_model()

@app.post("/predict")
async def predict_retina(
    file: UploadFile = File(...),
    model_type: str = Form("ml")  # Reads 'ml', 'dl', or 'transformer' from React
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Route 1: Machine Learning (Random Forest)
        if model_type == "ml":
            features = extract_ml_features(image)
            prediction, confidence = run_ml_inference(ml_model, features)
            return {
                "prediction": prediction,
                "confidence": confidence,
                "engine": "Random Forest (ML)"
            }

        # Route 2: Deep Learning (EfficientNet-B3)
        elif model_type == "dl":
            raise HTTPException(
                status_code=501, 
                detail="Deep Learning (EfficientNet) weights not found. Train Phase 2 model first."
            )

        # Route 3: Vision Transformer (ViT)
        elif model_type == "transformer":
            raise HTTPException(
                status_code=501, 
                detail="Vision Transformer weights not found. Train Phase 3 model first."
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid model type specified.")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import joblib
from rembg import remove
import io
from PIL import Image

# --- 1. INITIALIZE APP ---
app = FastAPI(title="BodyFat AI Estimator")

# Allow your Flutter app to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD MODELS (Global Variables) ---
print("Loading Models...")
try:
    # Use compile=False to fix the loss function bug
    model_A = tf.keras.models.load_model("best_model.h5", compile=False)
    model_neck = joblib.load("neck_model.joblib")
    model_bf = joblib.load("bodyfat_model.joblib")
    print("✅ Models Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # In production, you might want to exit here if models fail

# Constants
MEASUREMENT_COLUMNS = [
    'ankle', 'arm-length', 'bicep', 'calf', 'chest', 'forearm', 'height',
    'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch',
    'thigh', 'waist', 'wrist'
]
IMG_SIZE = 224

# --- 3. HELPER FUNCTIONS ---
def process_image_data(image_bytes):
    """
    Converts raw image bytes -> numpy array -> removes bg -> prepares tensor
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    
    # OpenCV uses BGR, PIL uses RGB. Convert if needed, but rembg handles RGB fine.
    # We'll stick to RGB for consistency with training.
    
    # Remove Background
    output = remove(image_np)
    
    # Extract Mask (Alpha channel > 50)
    # output is RGBA. Index 3 is Alpha.
    mask = (output[:, :, 3] > 50).astype(np.uint8) * 255
    
    # Prepare Tensor
    tensor = tf.convert_to_tensor(mask)
    tensor = tf.expand_dims(tensor, axis=-1)
    tensor = tf.image.grayscale_to_rgb(tensor)
    tensor = tf.image.resize_with_pad(tensor, IMG_SIZE, IMG_SIZE)
    tensor = tensor / 255.0 
    return tf.expand_dims(tensor, axis=0)

# --- 4. API ENDPOINT ---
@app.post("/predict")
async def predict_bodyfat(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    gender: str = Form(...),
    age: int = Form(...),
    height: float = Form(...),
    weight: float = Form(...)
):
    try:
        # A. Read and Process Images
        front_bytes = await front_image.read()
        side_bytes = await side_image.read()
        
        front_tensor = process_image_data(front_bytes)
        side_tensor = process_image_data(side_bytes)

        # B. Run Model A (Vision)
        raw_preds = model_A.predict([front_tensor, side_tensor], verbose=0)[0]
        measurements = dict(zip(MEASUREMENT_COLUMNS, raw_preds))

        # C. Scaling (Height Correction)
        # Prevent division by zero if model predicts 0 height (highly unlikely)
        pred_height = measurements['height'] if measurements['height'] > 0 else 170.0
        scaling_factor = height / pred_height
        
        scaled_waist = measurements['waist'] * scaling_factor
        scaled_hip = measurements['hip'] * scaling_factor
        scaled_chest = measurements['chest'] * scaling_factor

        # D. Run Neck Predictor
        kaggle_gender = "M" if gender.lower().startswith("m") else "F"
        
        neck_input = pd.DataFrame([{
            'Sex': kaggle_gender, 'Age': age, 'Weight': weight, 
            'Height': height, 'Chest': scaled_chest, 'Abdomen': scaled_waist
        }])
        
        predicted_neck = model_neck.predict(neck_input)[0]

        # E. Run AI Body Fat Model
        bf_input = pd.DataFrame([{
            'Sex': kaggle_gender, 'Age': age, 'Weight': weight, 
            'Height': height, 'Abdomen': scaled_waist, 'Hip': scaled_hip,
            'Neck': predicted_neck 
        }])
        
        ai_body_fat = model_bf.predict(bf_input)[0]

        # F. Run US Navy Method
        # Note: Formulas expect CM. Our scaled values are in CM.
        if kaggle_gender == "M":
            value = scaled_waist - predicted_neck
            if value <= 0: value = 1.0
            density = 1.0324 - 0.19077 * np.log10(value) + 0.15456 * np.log10(height)
            navy_body_fat = (495.0 / density) - 450.0
        else:
            value = scaled_waist + scaled_hip - predicted_neck
            if value <= 0: value = 1.0
            density = 1.29579 - 0.35004 * np.log10(value) + 0.22100 * np.log10(height)
            navy_body_fat = (495.0 / density) - 450.0

        # G. Return JSON Response
        return {
            "status": "success",
            "body_fat": {
                "ai_estimate": round(float(ai_body_fat), 1),
                "navy_estimate": round(float(navy_body_fat), 1),
                "average": round(float((ai_body_fat + navy_body_fat) / 2), 1)
            },
            "measurements": {
                "waist_cm": round(float(scaled_waist), 1),
                "hip_cm": round(float(scaled_hip), 1),
                "neck_cm": round(float(predicted_neck), 1),
                "chest_cm": round(float(scaled_chest), 1)
            },
            "meta": {
                "scaling_factor": round(float(scaling_factor), 4),
                "gender_used": kaggle_gender
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 5. RUN SERVER (If run directly) ---
if __name__ == "__main__":
    # Access via http://127.0.0.1:8000
    # To run: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
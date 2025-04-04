# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import uvicorn
import onnxruntime as ort

from utils import preprocess_image, class_names

# FastAPI app instance
app = FastAPI()

# Load ONNX model
onnx_model_path = "model/eyes_diseases.onnx"
ort_session = ort.InferenceSession(onnx_model_path)


@app.get("/")
def home():
    return {"message": "Welcome to the Eye Disease Detection API!"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(image.file).convert("RGB")
    input_tensor = preprocess_image(image)  # shape: (1, 224, 224, 3)

    # Run inference
    ort_inputs = {"input": input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions = ort_outs[0][0]  # Get first (and only) batch output

    # Get prediction
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100.0

    # Format result
    result = {
        "Predicted class": class_names[predicted_index],
        "Confidence": f"{confidence:.2f}%"
    }

    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

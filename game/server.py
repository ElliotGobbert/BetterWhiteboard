from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
from recognizer import SketchRecognizer

app = FastAPI()
recognizer = SketchRecognizer()

@app.get("/")
def get_ui():
    """Serves the simple localhost frontend."""
    with open("game/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict_drawing(file: UploadFile = File(...)):
    """API endpoint that receives the canvas image and returns a guess."""
    # Convert incoming file to numpy array, then to OpenCV image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
        
    guess, confidence = recognizer.predict(img)
    return {"guess": guess, "confidence": round(confidence * 100, 2)}

if __name__ == "__main__":
    # Run on all interfaces so the Jetson can be accessed from a laptop if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
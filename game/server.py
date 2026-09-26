from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import cv2
import numpy as np
import time
from recognizer import SketchRecognizer

app = FastAPI()
recognizer = SketchRecognizer()

# Global state to share between the OpenCV loop and the web frontend
app.state.latest_guess = "Waiting for drawing..."
app.state.latest_confidence = 0.0
app.state.latest_frame = None

@app.get("/")
def get_ui():
    with open("game/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict_drawing(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is not None:
        guess, confidence = recognizer.predict(img)
        app.state.latest_guess = guess
        app.state.latest_confidence = round(confidence * 100, 2)
        
    return {"guess": app.state.latest_guess, "confidence": app.state.latest_confidence}

@app.post("/update_frame")
async def update_frame(frame: UploadFile = File(...)):
    """Receives the live camera feed from main.py so the website can display it."""
    app.state.latest_frame = await frame.read()
    return {"status": "ok"}

@app.get("/latest_guess")
def get_latest_guess():
    return {"guess": app.state.latest_guess, "confidence": app.state.latest_confidence}

@app.get("/video_feed")
def video_feed():
    """Streams the latest OpenCV frame to the HTML image tag."""
    def iter_frames():
        while True:
            if app.state.latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + app.state.latest_frame + b'\r\n')
            time.sleep(0.05) # Cap at ~20 FPS to save CPU
    return StreamingResponse(iter_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
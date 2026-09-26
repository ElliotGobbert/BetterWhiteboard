import cv2
import numpy as np
# import onnxruntime as ort  # Uncomment when you add your ONNX model

class SketchRecognizer:
    def __init__(self, model_path="game/model.onnx"):
        # self.session = ort.InferenceSession(model_path)
        
        # Example categories - replace with your model's actual labels
        self.classes = ["apple", "banana", "cat", "dog", "house", "smiley face"] 

    def predict(self, canvas_img):
        """Processes the canvas and returns a prediction."""
        # 1. Preprocess: QuickDraw models usually expect 28x28 grayscale
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize pixel values (0 to 1)
        normalized = resized.astype(np.float32) / 255.0
        
        # Reshape for model input (Batch size 1, 1 channel, 28x28)
        input_data = np.expand_dims(normalized, axis=(0, 1))

        # 2. Run Inference (Mocked here until you drop in an ONNX model)
        # ort_inputs = {self.session.get_inputs()[0].name: input_data}
        # ort_outs = self.session.run(None, ort_inputs)
        # prediction_index = np.argmax(ort_outs[0])
        # return self.classes[prediction_index], float(ort_outs[0][0][prediction_index])
        
        # Mock return for testing the pipeline
        return "cat", 0.95
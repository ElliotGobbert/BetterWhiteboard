import os
import json
import cv2
import numpy as np
import onnxruntime as ort

class SketchRecognizer:
    def __init__(self, model_dir="game/models"):
        model_path = os.path.join(model_dir, "onnx", "model_quantized.onnx")
        config_path = os.path.join(model_dir, "config.json")

        self.model_loaded = False
        self.classes = {}

        if os.path.exists(model_path) and os.path.exists(config_path):
            try:
                self.session = ort.InferenceSession(model_path)
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.classes = config.get("id2label", {})
                self.model_loaded = True
                print("[INFO] ONNX QuickDraw model loaded successfully!")
            except Exception as e:
                print(f"[ERROR] Failed to initialize ONNX session: {e}")
        else:
            print("[WARNING] Model files missing. Run `uv run python game/download_model.py` first.")

    def predict(self, canvas_img):
        """Preprocesses the canvas and runs inference via ONNX Runtime."""
        if not self.model_loaded:
            return "Model Not Loaded", 0.0

        # Convert BGR canvas to RGB
        rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        
        # Scale pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose from (H, W, C) to (C, H, W) and add batch dimension -> (1, 3, 224, 224)
        chw = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        # Calculate Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        label = self.classes.get(str(best_idx), f"Class {best_idx}")

        return label, confidence
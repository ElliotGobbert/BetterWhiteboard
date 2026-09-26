from huggingface_hub import hf_hub_download
import os

def download_quickdraw_model():
    print("Downloading pre-trained QuickDraw model...")
    # Downloading a common lightweight QuickDraw ONNX model format
    # Note: Replace 'example-repo' with a specific QuickDraw ONNX repo if you decide to use a specific architecture like MobileNet.
    model_path = hf_hub_download(
        repo_id="nateraw/quickdraw-model", 
        filename="keras_metadata.pb", # Adjust filename based on the exact model format you choose to infer with (e.g., .onnx or .h5)
        local_dir="game/models"
    )
    print(f"Model downloaded successfully to: {model_path}")

if __name__ == "__main__":
    os.makedirs("game/models", exist_ok=True)
    download_quickdraw_model()
import os
from huggingface_hub import hf_hub_download

def download():
    print("[INFO] Downloading pre-trained QuickDraw ONNX model and label config...")
    os.makedirs("game/models", exist_ok=True)
    
    # Download quantized ONNX weights (~6 MB, optimized for Jetson/CPU)
    hf_hub_download(
        repo_id="Xenova/quickdraw-mobilevit-small",
        filename="onnx/model_quantized.onnx",
        local_dir="game/models"
    )
    
    # Download 345-class label mapping
    hf_hub_download(
        repo_id="Xenova/quickdraw-mobilevit-small",
        filename="config.json",
        local_dir="game/models"
    )
    print("[SUCCESS] Download complete! Model saved in game/models/")

if __name__ == "__main__":
    download()
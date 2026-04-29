import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from functools import lru_cache

MODEL_NAME = MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model():
    """
    Load model once and cache it in memory.
    On first run, downloads ~330MB weights from HuggingFace automatically.
    After that, loads from local cache instantly.
    """
    print(f"[INFO] Loading model: {MODEL_NAME}")
    print(f"[INFO] Using device: {DEVICE}")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    print("[INFO] Model loaded and ready!")
    return model, processor


def predict_image(image: Image.Image) -> dict:
    """
    Takes a PIL RGB image as input.
    Returns a dictionary with prediction result, confidence, and probabilities.

    Model labels:
        0 -> 'Fake'  (AI generated / manipulated face)  ✅ fixed comment
        1 -> 'Real'   (real authentic face)              ✅ fixed comment
    """
    model, processor = load_model()

    # Preprocess: resize to 224x224, normalize using ViT standard values
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    # Run model inference (no gradient needed, saves memory)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits          # raw scores shape: (1, 2)

    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    # Build label -> probability map using model's own config
    id2label = model.config.id2label    # {0: 'Fake', 1: 'Real'} ✅ fixed comment
    label_probs = {
        id2label[i]: round(probs[i], 4)
        for i in range(len(probs))
    }

    # Get the winning prediction
    predicted_idx = int(torch.argmax(logits, dim=1).item())
    predicted_label = id2label[predicted_idx]
    confidence = round(probs[predicted_idx], 4)

    # Normalize to simple FAKE / REAL for easy reading
    is_fake = "fake" in predicted_label.lower()  # ✅ fixed: was "deepfake"

    return {
        "result": "FAKE" if is_fake else "REAL",
        "confidence": confidence,
        "confidence_percent": f"{round(confidence * 100, 2)}%",
        "probabilities": label_probs,
        "model_used": MODEL_NAME,
        "device": str(DEVICE),
    }
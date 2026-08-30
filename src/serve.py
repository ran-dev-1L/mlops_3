import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from model import get_model

CHECKPOINT_PATH = Path("../checkpoints/classifier_v1.pt")
FASHION_MNIST_MEAN = [0.2860]
FASHION_MNIST_STD = [0.3530]
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

app = FastAPI(title="Fashion-MNIST Classifier")

model = None

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=FASHION_MNIST_MEAN, std=FASHION_MNIST_STD),
])


@app.on_event("startup")
def load_model():
    global model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    loaded_model = get_model(architecture="simple_cnn", num_classes=10)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    model = loaded_model


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = inference_transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    predicted_idx = int(torch.argmax(probabilities))

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "probabilities": {
            CLASS_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
    }
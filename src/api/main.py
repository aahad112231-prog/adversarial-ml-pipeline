"""
FastAPI backend serving the trained CIFAR-10 classifier.
Accepts an uploaded image, returns the predicted class + confidence.
"""

import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.models.cnn import SimpleCNN
from src.data.dataset import CIFAR10_CLASSES

app = FastAPI(title="CIFAR-10 Classifier API")

# Allow the frontend (served from a different origin/port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
model.eval()  # inference mode -- dropout off

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # in case the uploaded image isn't already 32x32
    transforms.ToTensor(),
])


@app.get("/")
def root():
    return {"status": "API is running", "model": "SimpleCNN", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    # Top 3 predictions, for a richer response
    top3_probs, top3_idx = torch.topk(probabilities, 3)
    top3 = [
        {"class": CIFAR10_CLASSES[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top3_probs, top3_idx)
    ]

    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "confidence": round(confidence, 4),
        "top3": top3,
    }

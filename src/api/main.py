"""
FastAPI backend serving the CIFAR-10 classifier with adversarial attack
and defense endpoints. Demonstrates the full adversarial ML pipeline.
"""

import io
import sys
import torch
import torch.nn.functional as F
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

sys.path.insert(0, '/app')
from src.models.cnn import SimpleCNN
from src.data.dataset import CIFAR10_CLASSES
from src.attacks.fgsm import fgsm_attack
from src.defenses.input_sanitization import jpeg_compress

app = FastAPI(title="Adversarial ML Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")

# Load baseline model
baseline_model = SimpleCNN().to(device)
baseline_model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
baseline_model.eval()

# Load robust model
robust_model = SimpleCNN().to(device)
robust_model.load_state_dict(torch.load("robust_cnn.pt", map_location=device))
robust_model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def tensor_to_base64(tensor):
    """Convert a [3, H, W] tensor to a base64 PNG string for sending to frontend."""
    pil_img = TF.to_pil_image(tensor.cpu().clamp(0, 1))
    pil_img = pil_img.resize((128, 128), Image.NEAREST)  # upscale for visibility
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict_single(model, tensor):
    """Run inference on a single image tensor [1, 3, H, W], return class + confidence + top3."""
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        top3_probs, top3_idx = torch.topk(probs, 3)
        top3 = [
            {"class": CIFAR10_CLASSES[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top3_probs, top3_idx)
        ]
    return CIFAR10_CLASSES[pred_idx], round(confidence, 4), top3


@app.get("/")
def root():
    return {"status": "Adversarial ML Pipeline API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), epsilon: float = 0.03):
    """
    Full pipeline analysis:
    1. Clean prediction (baseline model)
    2. FGSM attack prediction (baseline model on attacked image)
    3. Sanitization defense (baseline model on JPEG-compressed attacked image)
    4. Robust model prediction (adversarially trained model on attacked image)
    Returns images as base64 for frontend display.
    """
    # Load and preprocess image
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 1. Clean prediction
    clean_class, clean_conf, clean_top3 = predict_single(baseline_model, input_tensor)

    # 2. FGSM attack
    # We need a label to compute loss -- use the model's own clean prediction
    pseudo_label = torch.tensor([CIFAR10_CLASSES.index(clean_class)]).to(device)
    adv_tensor = fgsm_attack(baseline_model, input_tensor, pseudo_label, epsilon, device)
    adv_class, adv_conf, adv_top3 = predict_single(baseline_model, adv_tensor)

    # 3. JPEG sanitization defense on attacked image
    sanitized_tensor = jpeg_compress(adv_tensor, quality=75)
    san_class, san_conf, san_top3 = predict_single(baseline_model, sanitized_tensor)

    # 4. Robust model on attacked image
    robust_class, robust_conf, robust_top3 = predict_single(robust_model, adv_tensor)

    # Compute perturbation (amplified for visibility)
    perturbation = adv_tensor - input_tensor
    pert_amplified = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)

    return {
        "epsilon": epsilon,
        "images": {
            "clean":        tensor_to_base64(input_tensor[0]),
            "adversarial":  tensor_to_base64(adv_tensor[0]),
            "perturbation": tensor_to_base64(pert_amplified[0]),
            "sanitized":    tensor_to_base64(sanitized_tensor[0]),
        },
        "predictions": {
            "clean":      {"class": clean_class,   "confidence": clean_conf,   "top3": clean_top3},
            "adversarial":{"class": adv_class,     "confidence": adv_conf,     "top3": adv_top3},
            "sanitized":  {"class": san_class,     "confidence": san_conf,     "top3": san_top3},
            "robust":     {"class": robust_class,  "confidence": robust_conf,  "top3": robust_top3},
        }
    }

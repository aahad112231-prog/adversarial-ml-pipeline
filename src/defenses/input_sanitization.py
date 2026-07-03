"""
Input sanitization defense via JPEG compression.

Compresses input images through JPEG encoding/decoding before passing
them to the model, which destroys much of the carefully crafted
adversarial perturbation while preserving enough of the image for
correct classification.
"""

import io
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def jpeg_compress(images, quality=75):
    """
    Applies JPEG compression to a batch of tensors.

    Args:
        images: tensor of shape [B, 3, H, W], values in [0, 1]
        quality: JPEG quality (1-95). Lower = more compression = more perturbation destroyed,
                 but also more image quality lost. 75 is a good starting balance.

    Returns:
        compressed: same shape as input, values in [0, 1]
    """
    compressed = []
    for img in images:
        # Convert tensor [3, H, W] → PIL image
        pil_img = TF.to_pil_image(img.cpu())

        # Compress to JPEG in memory (never actually writes to disk)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Decompress back to tensor
        decompressed = Image.open(buffer).convert("RGB")
        tensor = TF.to_tensor(decompressed)
        compressed.append(tensor)

    return torch.stack(compressed).to(images.device)


if __name__ == "__main__":
    import torch
    from src.models.cnn import SimpleCNN
    from src.data.dataset import get_dataloaders
    from src.attacks.fgsm import fgsm_attack
    from src.attacks.pgd import pgd_attack

    device = torch.device("cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=64)

    clean_correct = 0
    fgsm_correct = 0
    fgsm_sanitized_correct = 0
    total = 0

    print("Evaluating input sanitization defense (JPEG quality=75)...\n")

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_preds = torch.argmax(model(images), dim=1)
            clean_correct += (clean_preds == labels).sum().item()

        # FGSM accuracy (no defense)
        adv_images = fgsm_attack(model, images, labels, epsilon=0.03, device=device)
        with torch.no_grad():
            adv_preds = torch.argmax(model(adv_images), dim=1)
            fgsm_correct += (adv_preds == labels).sum().item()

        # FGSM + sanitization
        sanitized = jpeg_compress(adv_images, quality=75)
        with torch.no_grad():
            san_preds = torch.argmax(model(sanitized), dim=1)
            fgsm_sanitized_correct += (san_preds == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100 * clean_correct / total
    fgsm_acc = 100 * fgsm_correct / total
    fgsm_san_acc = 100 * fgsm_sanitized_correct / total

    print(f"Clean accuracy (no defense):         {clean_acc:.2f}%")
    print(f"FGSM accuracy (no defense):          {fgsm_acc:.2f}%")
    print(f"FGSM accuracy (+ JPEG sanitization): {fgsm_san_acc:.2f}%")
    print(f"\nSanitization recovery: +{fgsm_san_acc - fgsm_acc:.2f} percentage points")

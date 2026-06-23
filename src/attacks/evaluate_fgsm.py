"""
Evaluates FGSM attack across the full CIFAR-10 test set.
Reports clean accuracy vs. adversarial accuracy at a given epsilon.
"""

import torch

from src.models.cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from src.attacks.fgsm import fgsm_attack


def evaluate_fgsm_full(epsilon=0.03, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=batch_size)

    clean_correct = 0
    adv_correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            clean_correct += (clean_preds == labels).sum().item()

        # Adversarial accuracy
        adv_images = fgsm_attack(model, images, labels, epsilon, device)
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            adv_correct += (adv_preds == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total

    print(f"\nEpsilon: {epsilon}")
    print(f"Clean accuracy:       {clean_acc:.2f}%")
    print(f"Adversarial accuracy: {adv_acc:.2f}%")
    print(f"Accuracy drop:        {clean_acc - adv_acc:.2f} percentage points")

    return clean_acc, adv_acc


if __name__ == "__main__":
    evaluate_fgsm_full(epsilon=0.03)

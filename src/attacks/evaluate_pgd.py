"""
Evaluates PGD attack across the full CIFAR-10 test set.
"""

import torch

from src.models.cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from src.attacks.pgd import pgd_attack


def evaluate_pgd_full(epsilon=0.03, alpha=0.01, num_steps=10, batch_size=64):
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

        with torch.no_grad():
            clean_preds = torch.argmax(model(images), dim=1)
            clean_correct += (clean_preds == labels).sum().item()

        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps, device)
        with torch.no_grad():
            adv_preds = torch.argmax(model(adv_images), dim=1)
            adv_correct += (adv_preds == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total

    print(f"\nEpsilon: {epsilon}, Alpha: {alpha}, Steps: {num_steps}")
    print(f"Clean accuracy:       {clean_acc:.2f}%")
    print(f"PGD adversarial accuracy: {adv_acc:.2f}%")
    print(f"Accuracy drop:        {clean_acc - adv_acc:.2f} percentage points")

    return clean_acc, adv_acc


if __name__ == "__main__":
    evaluate_pgd_full(epsilon=0.03, alpha=0.01, num_steps=10)

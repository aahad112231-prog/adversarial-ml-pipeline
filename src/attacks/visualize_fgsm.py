"""
Visualizes a clean image vs. its FGSM-perturbed counterpart, side by side.
Saves the result as a PNG since we're running headless (no display) in Docker.
"""

import torch
import matplotlib.pyplot as plt

from src.models.cnn import SimpleCNN
from src.data.dataset import get_dataloaders, CIFAR10_CLASSES
from src.attacks.fgsm import fgsm_attack


def visualize_attack(epsilon=0.03, num_examples=4, save_path="fgsm_comparison.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=num_examples)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        clean_preds = torch.argmax(model(images), dim=1)

    adv_images = fgsm_attack(model, images, labels, epsilon, device)

    with torch.no_grad():
        adv_preds = torch.argmax(model(adv_images), dim=1)

    # The actual perturbation, amplified so it's visible (it's normally near-invisible)
    perturbation = (adv_images - images)
    perturbation_visible = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())

    fig, axes = plt.subplots(3, num_examples, figsize=(3 * num_examples, 9))

    for i in range(num_examples):
        clean_img = images[i].cpu().permute(1, 2, 0).numpy()
        adv_img = adv_images[i].cpu().permute(1, 2, 0).numpy()
        pert_img = perturbation_visible[i].cpu().permute(1, 2, 0).numpy()

        true_label = CIFAR10_CLASSES[labels[i].item()]
        clean_label = CIFAR10_CLASSES[clean_preds[i].item()]
        adv_label = CIFAR10_CLASSES[adv_preds[i].item()]

        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f"Clean\nTrue: {true_label}\nPred: {clean_label}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(pert_img)
        axes[1, i].set_title("Perturbation\n(amplified to be visible)", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(adv_img)
        color = "red" if adv_label != true_label else "green"
        axes[2, i].set_title(f"Adversarial\nPred: {adv_label}", fontsize=9, color=color)
        axes[2, i].axis("off")

    plt.suptitle(f"FGSM Attack (epsilon={epsilon})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    visualize_attack(epsilon=0.03)

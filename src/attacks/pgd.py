"""
PGD (Projected Gradient Descent) adversarial attack.

A stronger, iterative version of FGSM: takes several small gradient-ascent
steps instead of one big one, projecting (clipping) the perturbation back
into the allowed epsilon-ball after every step.

Reference: Madry et al., 2017, "Towards Deep Learning Models Resistant to
Adversarial Attacks"
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device):
    """
    Generates adversarial examples using PGD.

    Args:
        model: the trained model to attack
        images: batch of input images, shape [B, 3, H, W], values in [0,1]
        labels: true labels for the images, shape [B]
        epsilon: total perturbation budget (max allowed change per pixel)
        alpha: step size for each individual iteration (smaller than epsilon)
        num_steps: how many gradient ascent steps to take
        device: cpu or cuda

    Returns:
        perturbed_images: adversarial examples, same shape as images, clipped to [0,1]
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Start from the original image -- this is what we'll iteratively perturb
    perturbed_images = images.clone().detach()

    for step in range(num_steps):
        perturbed_images.requires_grad = True

        outputs = model(perturbed_images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # One small FGSM-style step
            gradient_sign = perturbed_images.grad.sign()
            perturbed_images = perturbed_images + alpha * gradient_sign

            # PROJECTION step: keep total perturbation within epsilon of the ORIGINAL image
            perturbation = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
            perturbed_images = images + perturbation

            # Keep it a valid image
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

        perturbed_images = perturbed_images.detach()

    return perturbed_images


if __name__ == "__main__":
    from src.models.cnn import SimpleCNN
    from src.data.dataset import get_dataloaders, CIFAR10_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        clean_preds = torch.argmax(model(images), dim=1)

    epsilon = 0.03
    alpha = 0.01
    num_steps = 10
    adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_steps, device)

    with torch.no_grad():
        adv_preds = torch.argmax(model(adv_images), dim=1)

    print(f"Epsilon: {epsilon}, Alpha: {alpha}, Steps: {num_steps}\n")
    for i in range(len(labels)):
        true_label = CIFAR10_CLASSES[labels[i].item()]
        clean_pred = CIFAR10_CLASSES[clean_preds[i].item()]
        adv_pred = CIFAR10_CLASSES[adv_preds[i].item()]
        flipped = "FLIPPED" if clean_pred != adv_pred else "same"
        print(f"True: {true_label:12s} | Clean pred: {clean_pred:12s} | PGD pred: {adv_pred:12s} | {flipped}")

    clean_acc = (clean_preds == labels).float().mean().item() * 100
    adv_acc = (adv_preds == labels).float().mean().item() * 100
    print(f"\nClean accuracy on this batch: {clean_acc:.1f}%")
    print(f"PGD adversarial accuracy on this batch: {adv_acc:.1f}%")

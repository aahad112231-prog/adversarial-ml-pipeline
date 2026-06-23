"""
FGSM (Fast Gradient Sign Method) adversarial attack.

Computes a small perturbation that, added to an input image, maximizes
the model's loss -- causing misclassification -- while keeping the
perturbation small enough to be (ideally) imperceptible to a human.

Reference: Goodfellow et al., 2014, "Explaining and Harnessing
Adversarial Examples"
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, device):
    """
    Generates adversarial examples using FGSM.

    Args:
        model: the trained model to attack
        images: batch of input images, shape [B, 3, H, W], values in [0,1]
        labels: true labels for the images, shape [B]
        epsilon: perturbation size (how much each pixel can change)
        device: cpu or cuda

    Returns:
        perturbed_images: adversarial examples, same shape as images, clipped to [0,1]
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    # Tell PyTorch to track gradients with respect to the IMAGE, not just the weights
    images.requires_grad = True

    outputs = model(images)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    model.zero_grad()      # clear any stale gradients
    loss.backward()        # backprop all the way to the input image

    # The actual FGSM step: move the image in the direction that INCREASES loss
    gradient_sign = images.grad.sign()
    perturbed_images = images + epsilon * gradient_sign

    # Keep the result a valid image
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images.detach()  # detach -- we're done tracking gradients now


if __name__ == "__main__":
    # Quick sanity check using a few real CIFAR-10 test images
    from src.models.cnn import SimpleCNN
    from src.data.dataset import get_dataloaders, CIFAR10_CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=8)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Predictions BEFORE attack
    with torch.no_grad():
        clean_outputs = model(images)
        clean_preds = torch.argmax(clean_outputs, dim=1)

    # Generate adversarial examples
    epsilon = 0.03
    adv_images = fgsm_attack(model, images, labels, epsilon, device)

    # Predictions AFTER attack
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = torch.argmax(adv_outputs, dim=1)

    print(f"Epsilon: {epsilon}\n")
    for i in range(len(labels)):
        true_label = CIFAR10_CLASSES[labels[i].item()]
        clean_pred = CIFAR10_CLASSES[clean_preds[i].item()]
        adv_pred = CIFAR10_CLASSES[adv_preds[i].item()]
        flipped = "FLIPPED" if clean_pred != adv_pred else "same"
        print(f"True: {true_label:12s} | Clean pred: {clean_pred:12s} | Adversarial pred: {adv_pred:12s} | {flipped}")

    clean_acc = (clean_preds == labels).float().mean().item() * 100
    adv_acc = (adv_preds == labels).float().mean().item() * 100
    print(f"\nClean accuracy on this batch: {clean_acc:.1f}%")
    print(f"Adversarial accuracy on this batch: {adv_acc:.1f}%")

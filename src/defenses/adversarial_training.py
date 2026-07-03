"""
Adversarial training defense.

Retrains the CNN on a mix of clean + adversarially perturbed images,
forcing the model to learn robustness against attacks during training itself.

Based on the approach from Madry et al., 2017.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src.models.cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from src.attacks.fgsm import fgsm_attack


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def evaluate_adversarial(model, loader, epsilon, device):
    """Evaluate model accuracy under FGSM attack."""
    from src.attacks.fgsm import fgsm_attack
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(model, images, labels, epsilon, device)
        with torch.no_grad():
            preds = torch.argmax(model(adv_images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def adversarial_train(epochs=10, learning_rate=0.001, batch_size=64, epsilon=0.03):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Adversarial training with epsilon={epsilon}\n")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run(run_name="adversarial_training"):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("defense", "adversarial_training_fgsm")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Generate adversarial examples for this batch
                adv_images = fgsm_attack(model, images, labels, epsilon, device)

                # Train on BOTH clean and adversarial images
                combined_images = torch.cat([images, adv_images], dim=0)
                combined_labels = torch.cat([labels, labels], dim=0)

                optimizer.zero_grad()
                outputs = model(combined_images)
                loss = criterion(outputs, combined_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            clean_acc = evaluate(model, test_loader, device)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Clean Acc: {clean_acc:.2f}%")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("clean_accuracy", clean_acc, step=epoch)

        # Final evaluation against both attacks
        print("\nFinal evaluation:")
        clean_acc = evaluate(model, test_loader, device)
        fgsm_acc = evaluate_adversarial(model, test_loader, epsilon, device)

        print(f"Clean accuracy:       {clean_acc:.2f}%")
        print(f"FGSM accuracy:        {fgsm_acc:.2f}%")

        mlflow.log_metric("final_clean_accuracy", clean_acc)
        mlflow.log_metric("final_fgsm_accuracy", fgsm_acc)

        torch.save(model.state_dict(), "robust_cnn.pt")
        mlflow.log_artifact("robust_cnn.pt")
        print("\nRobust model saved to robust_cnn.pt")


if __name__ == "__main__":
    adversarial_train(epochs=10, learning_rate=0.001, batch_size=64, epsilon=0.03)

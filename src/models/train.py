"""
Trains SimpleCNN on CIFAR-10, tracking the run with MLflow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src.data.dataset import get_dataloaders
from src.models.cnn import SimpleCNN


def evaluate(model, test_loader, device):
    """Runs the model on the test set, returns accuracy as a percentage."""
    model.eval()  # switches off dropout etc. -- we want deterministic behavior when evaluating
    correct = 0
    total = 0
    with torch.no_grad():  # no need to track gradients, we're not training here
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # pick the highest-scoring class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train(epochs=10, learning_rate=0.001, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run():
        # Log the hyperparameters for this run
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model", "SimpleCNN")

        for epoch in range(epochs):
            model.train()  # switches dropout back on -- we want regularization during training
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()        # clear gradients from the previous batch
                outputs = model(images)      # forward pass
                loss = criterion(outputs, labels)
                loss.backward()              # compute gradients
                optimizer.step()             # update weights

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            test_accuracy = evaluate(model, test_loader, device)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

            # Log metrics for this epoch -- this is what lets you see a curve over time in the dashboard
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

        # Save the final trained model
        torch.save(model.state_dict(), "baseline_cnn.pt")
        mlflow.log_artifact("baseline_cnn.pt")
        print("Model saved to baseline_cnn.pt")


if __name__ == "__main__":
    train(epochs=10, learning_rate=0.001, batch_size=64)

"""
Loads CIFAR-10 and prepares PyTorch DataLoaders.
Images are kept in [0, 1] range (no mean/std normalization) so that
adversarial perturbations later have a simple, interpretable scale.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_dataloaders(data_dir="./data", batch_size=64, num_workers=2):
    """
    Downloads CIFAR-10 (if not already present) and returns
    train and test DataLoaders.

    Args:
        data_dir: where to store/look for the dataset
        batch_size: number of images per training batch
        num_workers: parallel subprocesses for data loading

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts PIL image -> tensor, scales 0-255 to 0-1 automatically
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Pixel value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels in batch: {labels[:10].tolist()}")

"""
Robustness gate test -- runs as part of CI/CD pipeline.
Fails (exit code 1) if adversarial accuracy drops below threshold.
This is the automated security check that gates model deployment.
"""

import sys
import torch
from src.models.cnn import SimpleCNN
from src.data.dataset import get_dataloaders
from src.attacks.fgsm import fgsm_attack

# Robustness thresholds -- if adversarial accuracy falls below these,
# the pipeline fails and the model is NOT approved for deployment
CLEAN_ACCURACY_THRESHOLD = 60.0   # % -- minimum acceptable clean accuracy
FGSM_ACCURACY_THRESHOLD = 10.0    # % -- minimum acceptable robustness under attack

EPSILON = 0.03
NUM_BATCHES = 10  # evaluate on 10 batches (~640 images) for speed in CI


def run_robustness_check(model_path="baseline_cnn.pt"):
    print(f"Loading model from: {model_path}")
    print(f"Robustness gate thresholds:")
    print(f"  Clean accuracy  >= {CLEAN_ACCURACY_THRESHOLD}%")
    print(f"  FGSM accuracy   >= {FGSM_ACCURACY_THRESHOLD}% (epsilon={EPSILON})")
    print()

    device = torch.device("cpu")  # CI always runs on CPU

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders(batch_size=64)

    clean_correct = 0
    adv_correct = 0
    total = 0

    for i, (images, labels) in enumerate(test_loader):
        if i >= NUM_BATCHES:
            break

        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = torch.argmax(model(images), dim=1)
            clean_correct += (clean_preds == labels).sum().item()

        adv_images = fgsm_attack(model, images, labels, EPSILON, device)
        with torch.no_grad():
            adv_preds = torch.argmax(model(adv_images), dim=1)
            adv_correct += (adv_preds == labels).sum().item()

        total += labels.size(0)

    clean_acc = 100 * clean_correct / total
    adv_acc = 100 * adv_correct / total

    print(f"Results on {total} test images:")
    print(f"  Clean accuracy:  {clean_acc:.2f}%")
    print(f"  FGSM accuracy:   {adv_acc:.2f}%")
    print()

    # Gate checks
    passed = True

    if clean_acc < CLEAN_ACCURACY_THRESHOLD:
        print(f"FAILED: Clean accuracy {clean_acc:.2f}% is below threshold {CLEAN_ACCURACY_THRESHOLD}%")
        passed = False
    else:
        print(f"PASSED: Clean accuracy {clean_acc:.2f}% >= {CLEAN_ACCURACY_THRESHOLD}%")

    if adv_acc < FGSM_ACCURACY_THRESHOLD:
        print(f"FAILED: FGSM accuracy {adv_acc:.2f}% is below threshold {FGSM_ACCURACY_THRESHOLD}%")
        passed = False
    else:
        print(f"PASSED: FGSM accuracy {adv_acc:.2f}% >= {FGSM_ACCURACY_THRESHOLD}%")

    print()
    if passed:
        print("ROBUSTNESS GATE: PASSED -- model approved")
        sys.exit(0)
    else:
        print("ROBUSTNESS GATE: FAILED -- model rejected")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline_cnn.pt", help="Path to model checkpoint")
    args = parser.parse_args()
    run_robustness_check(model_path=args.model)

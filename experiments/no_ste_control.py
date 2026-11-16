
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import binarize
from model import BNN

SEED = 577
EPOCHS = 10
BATCH_SIZE = 100
LR = 0.001

def run_experiment(disable_ste):
    torch.manual_seed(SEED)
    binarize.ste_off = disable_ste

    train_data = datasets.MNIST(
        root="./data", train=True, download=True,
        transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = BNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    results = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer, nn.Linear):
                        binarize.clip_weights(layer.weight)

            total_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / len(test_data)
        avg_loss = total_loss / len(train_data)
        results.append((epoch + 1, avg_loss, accuracy))

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%",
            flush=True
        )

    return results


if __name__ == "__main__":
    print("\nExperiment 1: WITH STE", flush=True)
    with_ste = run_experiment(False)

    print("\nExperiment 2: WITHOUT STE", flush=True)
    without_ste = run_experiment(True)

    import csv
    import matplotlib.pyplot as plt

    output_dir = Path(__file__).resolve().parent

    with open(output_dir / "no_ste_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "with_ste_loss", "with_ste_accuracy",
                         "without_ste_loss", "without_ste_accuracy"])
        for a, b in zip(with_ste, without_ste):
            writer.writerow([a[0], a[1], a[2], b[1], b[2]])

    epochs = [r[0] for r in with_ste]

    for metric, index in [("accuracy", 2), ("loss", 1)]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [r[index] for r in with_ste],
                 marker="o", label="With STE")
        plt.plot(epochs, [r[index] for r in without_ste],
                 marker="s", label="Without STE")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy (%)" if metric == "accuracy" else "Training Loss")
        plt.title(f"BNN: With STE vs Without STE - {metric.title()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"no_ste_{metric}.png", dpi=200)
        plt.close()

    print("\nFINAL COMPARISON")
    print(f"With STE: {with_ste[-1][2]:.2f}%")
    print(f"Without STE: {without_ste[-1][2]:.2f}%")
    print("CSV and graphs saved.")

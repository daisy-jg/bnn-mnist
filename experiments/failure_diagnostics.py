
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import binarize
from model import BNN

SEED = 577
EPOCHS = 10
BATCH_SIZE = 100
LR = 0.001
OUTPUT_DIR = Path(__file__).resolve().parent


def run_diagnostics(disable_ste):
    torch.manual_seed(SEED)
    binarize.ste_off = disable_ste

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )

    model = BNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    results = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        gradient_sum = 0
        weight_change_sum = 0
        batches = 0

        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Measure gradient of first binarized layer
            grad = model.fc1.weight.grad
            grad_norm = grad.norm().item() if grad is not None else 0.0
            gradient_sum += grad_norm

            old_weights = model.fc1.weight.detach().clone()

            optimizer.step()

            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer, nn.Linear):
                        binarize.clip_weights(layer.weight)

                weight_change = (
                    model.fc1.weight - old_weights
                ).abs().mean().item()

            weight_change_sum += weight_change
            total_loss += loss.item() * images.size(0)
            batches += 1

        avg_loss = total_loss / len(train_data)
        avg_grad = gradient_sum / batches
        avg_change = weight_change_sum / batches

        results.append((epoch + 1, avg_loss, avg_grad, avg_change))

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={avg_loss:.4f}, "
            f"Gradient={avg_grad:.6f}, "
            f"Weight change={avg_change:.8f}",
            flush=True
        )

    return results


if __name__ == "__main__":
    print("\nExperiment 3A: WITH STE", flush=True)
    with_ste = run_diagnostics(False)

    print("\nExperiment 3B: WITHOUT STE", flush=True)
    without_ste = run_diagnostics(True)

    with open(OUTPUT_DIR / "failure_diagnostics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "with_ste_loss", "with_ste_gradient",
            "with_ste_weight_change", "no_ste_loss",
            "no_ste_gradient", "no_ste_weight_change"
        ])
        for a, b in zip(with_ste, without_ste):
            writer.writerow([a[0], a[1], a[2], a[3],
                             b[1], b[2], b[3]])

    epochs = [row[0] for row in with_ste]

    for name, index, ylabel in [
        ("gradient", 2, "Average Gradient Norm"),
        ("weight_change", 3, "Average Absolute Weight Change")
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [r[index] for r in with_ste],
                 marker="o", label="With STE")
        plt.plot(epochs, [r[index] for r in without_ste],
                 marker="s", label="Without STE")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"BNN Failure Diagnostics: {name.replace('_', ' ').title()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"failure_{name}.png", dpi=200)
        plt.close()

    print("\nFINAL DIAGNOSTICS")
    print(f"With STE gradient: {with_ste[-1][2]:.6f}")
    print(f"Without STE gradient: {without_ste[-1][2]:.6f}")
    print(f"With STE weight change: {with_ste[-1][3]:.8f}")
    print(f"Without STE weight change: {without_ste[-1][3]:.8f}")
    print("CSV and graphs saved.")

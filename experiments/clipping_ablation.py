
# CS 577 - BNN MNIST
# Ahlam: Weight Clipping Ablation
# Compare BNN training with and without weight clipping.

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import binarize
from model import BNN
from binarize import clip_weights


# Experiment settings
SEED = 577
EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# Keep STE enabled in both experiments
binarize.ste_off = False

# Save results in the experiments folder
RESULTS_DIR = Path(__file__).resolve().parent


def apply_clipping(model):
    # Clip real-valued weights after optimizer.step()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                clip_weights(layer.weight)


def run_experiment(use_clipping):
    # Same random seed for fair comparison
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_fn = nn.CrossEntropyLoss()

    history = []
    max_weight_seen = 0.0
    clipping_count = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            # Count weights outside [-1, 1] before clipping
            if use_clipping:
                with torch.no_grad():
                    for layer in model.modules():
                        if isinstance(layer, nn.Linear):
                            clipping_count += (
                                (layer.weight.abs() > 1.0)
                                .sum()
                                .item()
                            )

                apply_clipping(model)

            # Track maximum absolute weight after optional clipping
            with torch.no_grad():
                current_max = max(
                    layer.weight.abs().max().item()
                    for layer in model.modules()
                    if isinstance(layer, nn.Linear)
                )
                max_weight_seen = max(
                    max_weight_seen,
                    current_max
                )

            # Accumulate loss for every training batch
            total_loss += loss.item() * images.size(0)

        # Evaluate on MNIST test set
        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (
                    (predictions == labels).sum().item()
                )

        accuracy = 100 * correct / len(test_data)
        avg_loss = total_loss / len(train_data)

        history.append((epoch + 1, avg_loss, accuracy))

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={avg_loss:.4f}, "
            f"Accuracy={accuracy:.2f}%"
        )

    print(
        f"Maximum absolute weight during training: "
        f"{max_weight_seen:.4f}"
    )

    if use_clipping:
        print(
            f"Weights requiring clipping (total occurrences): "
            f"{clipping_count}"
        )

    return history, max_weight_seen


def save_results(clipping_results, no_clipping_results):
    csv_path = RESULTS_DIR / "clipping_results.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Experiment", "Epoch", "Loss", "Accuracy"]
        )

        for epoch, loss, accuracy in clipping_results:
            writer.writerow(
                ["With Clipping", epoch, loss, accuracy]
            )

        for epoch, loss, accuracy in no_clipping_results:
            writer.writerow(
                ["Without Clipping", epoch, loss, accuracy]
            )

    print(f"\nResults saved to {csv_path}")


def create_graphs(clipping_results, no_clipping_results):
    epochs = [row[0] for row in clipping_results]

    clip_loss = [row[1] for row in clipping_results]
    no_clip_loss = [row[1] for row in no_clipping_results]

    clip_acc = [row[2] for row in clipping_results]
    no_clip_acc = [row[2] for row in no_clipping_results]

    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, clip_acc, marker="o", label="With Clipping")
    plt.plot(
        epochs, no_clip_acc,
        marker="s", linestyle="--",
        label="Without Clipping"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("BNN: Clipping vs No Clipping - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "clipping_accuracy.png", dpi=200)
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, clip_loss, marker="o", label="With Clipping")
    plt.plot(
        epochs, no_clip_loss,
        marker="s", linestyle="--",
        label="Without Clipping"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("BNN: Clipping vs No Clipping - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "clipping_loss.png", dpi=200)
    plt.close()

    print("Graphs saved successfully.")


if __name__ == "__main__":
    print("\nExperiment 1: WITH clipping")
    clipping_results, clipping_max = run_experiment(True)

    print("\nExperiment 2: WITHOUT clipping")
    no_clipping_results, no_clipping_max = run_experiment(False)

    save_results(clipping_results, no_clipping_results)
    create_graphs(clipping_results, no_clipping_results)

    print("\nFINAL COMPARISON")
    print(
        f"With clipping: "
        f"{clipping_results[-1][2]:.2f}% accuracy, "
        f"max weight={clipping_max:.4f}"
    )
    print(
        f"Without clipping: "
        f"{no_clipping_results[-1][2]:.2f}% accuracy, "
        f"max weight={no_clipping_max:.4f}"
    )

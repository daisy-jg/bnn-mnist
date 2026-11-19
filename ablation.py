import subprocess
import csv
import os
import sys

# Set random seeds
SEEDS = [25, 30, 35]
# Number of training epochs
EPOCHS = 10
# Create results folder to store results of completed runs
os.makedirs("results", exist_ok = True)
# Create logs folder to store logs of each run
os.makedirs("logs", exist_ok = True)

results = []
# Loop full precision and bnn model
for model_name in ["full", "bnn"]:
    for seed in SEEDS:
        print(f"\nModel: {model_name}, Seed: {seed}")

        # Terminal command string
        cmd = [
            sys.executable, "train_model.py",
            "--model", model_name,
            "--epochs", str(EPOCHS),
            "--seed", str(seed),
        ]

        # File path to save logs
        log_path = f"logs/{model_name}_seed{seed}.log"
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, stdout = f, stderr = subprocess.STDOUT, text = True)
        # Error message for failed run
        if proc.returncode != 0:
            print(f"Failed run")
            continue

        # Store accuracy value, training time, and parameters count
        final_acc = None
        train_time = None
        n_params = None
        with open(log_path) as f:
            # Get values for test accuracy, training time, and parameters
            for line in f:
                if "Test Accuracy:" in line:
                    final_acc = float(line.split("Test Accuracy:")[1].strip().rstrip("%"))
                elif "Total training time:" in line:
                    train_time = float(line.split(":")[1].strip().rstrip("s"))
                elif "Parameters:" in line:
                    n_params = int(line.split(":")[1].strip())

        if final_acc is None:
            print("Accuracy value is missing in log file")
            continue

        # Add dictionary of run's info to results list
        results.append({
            "model": model_name,
            "seed": seed,
            "final_accuracy": final_acc,
            "train_time_s": train_time,
            "n_params": n_params,
        })

        # Print final accuracy, training time, and parameters
        print(f"Final accuracy: {final_acc:.2f}% | Training Time: {train_time:.2f}s | Parameters: {n_params}")

# Save results to CSV file after all runs have completed
with open("results/ablation_accuracy.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model", "seed", "final_accuracy", "train_time_s", "n_params"]
    )
    writer.writeheader()
    writer.writerows(results)
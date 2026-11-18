"""
Implements the following:

1. Loads dataset (MNIST), trains, and evaluates model

2. Contains command-line switches (--clip, --no-ste) so Ahlam and Daisy 
can run their experiments without rewriting the hardcode
"""

# Import libraries:
import argparse
import torch
import torch.nn as nn
from binarize import clip_weights
from model import BNN, FullPrecision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import binarize
import random
import numpy as np
import time

# Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load dataset (MNIST):
t = transforms.ToTensor()

# Load dataset (MNIST):
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=t)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=t)

# Split training and test data:
train_set = DataLoader(train_data, batch_size=100, shuffle=True)
test_set= DataLoader(test_data, batch_size=100, shuffle=False)

# Evaluate model:
def evaluate(model):
    model.eval()

    # Counter for True Pos:
    correct_pred = 0

    # evaluate without calculating gradients yet (speed and memory purposes):
    with torch.no_grad():
        for input,label in test_set:
            # Get outputs:
            output = model(input)

            # Get predicted class:
            pred_class = output.argmax(dim=1)

            #  Update number of correct predictions:
            correct_pred += (pred_class == label).sum().item()
    return correct_pred/len(test_data)

# Parser and switches for teammates to run experiments:
parser = argparse.ArgumentParser()

# "--" means optional arg:
parser.add_argument('--model', type=str, default='bnn')
parser.add_argument('--no_ste', action='store_true')
parser.add_argument('--no_clip', action='store_true')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type = int, default = 25)
args = parser.parse_args()
set_seed(args.seed)

# If STE is off, updated args:
binarize.ste_off = args.no_ste

# If statement for running either model:
if args.model == 'bnn':
    model = BNN()
else:
    model = FullPrecision()

# Update model weights using the grad:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss func:
cross_entropy = nn.CrossEntropyLoss()

# Track training time and display parameters
start_time = time.time()
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params}")

# Train Model:
for epoch in range(args.epochs):
    # Switch to training the model:
    model.train()

    # Iterate through training set:
    for input,label in train_set:
        # Reset grad to 0:
        optimizer.zero_grad()

        # Forward pass:
        output = model(input)

        # Loss calculation:
        loss_value = cross_entropy(output,label)

        # Backward pass:
        loss_value.backward()

        # Ablation (weight clipping):
        if args.model == "bnn" and not args.no_clip:
            for m in model.modules():
                # check if layer is linear and clip the weights:
                if isinstance(m, nn.Linear):
                    m.weight.data = clip_weights(m.weight.data)

        # Update weights:
        optimizer.step()

    # Evaluiate model after loop:
    accuracy = evaluate(model)

    # Display accuracy:
    print(f"Epoch {epoch+1}, Test Accuracy: {accuracy*100:.2f}%") 

# Display training time
elapsed = time.time() - start_time
print(f"Total training time: {elapsed:.2f}s")
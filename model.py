"""
Implements the following:

1. BNN (Binarized Neural Network) - It binarizes both the weights and activations in the forward pass

2. FullPrecision: The baseline model for Daisy's role. Apart from having the same architecture as the BNN, 
it instead uses 32-bit floats and ReLU activations (basically a regular NN)
"""
# Import library:
import torch.nn as nn
from binarize import binarize

# Binarized LL:
class BinLinearLayer(nn.Linear):
    # Linear Layer that binarizes w in the forward pass
    def forward(self, x):
        # Calls the STE class in binarize.py
        binarized_w = binarize(self.weight)

        # Calculate linear layer:
        return nn.functional.linear(x,binarized_w,self.bias)

# Binarized NN:
class BNN(nn.Module):
    #  Constructor:
    def __init__(self, input_dim = 784, hidden_dim = 256, output_dim = 10):
        super().__init__()

        # Layer 1 (fully connected layer):
        self.fc1 = BinLinearLayer(input_dim, hidden_dim)

        # Normalization:
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2:
        self.fc2 = BinLinearLayer(hidden_dim, output_dim)

    # Forward pass:
    def forward(self, x):
        # Flatten input:
        x = x.view(x.size(0), -1)

        # First pass:
        x = self.fc1(x)

        # Normalization:
        x = self.bn1(x)

        # Binarize the result:
        x = binarize(x)

        # Second pass:
        return self.fc2(x)

# Baseline class for comparison- Regular NN model:
class FullPrecision(nn.Module):
    #  Constructor:
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        
        # NN Linear Layer:
        self.net = nn.Sequential(

            # Flattens input:
            nn.Flatten(),

            # Linear Layer:             
            nn.Linear(input_dim, hidden_dim),

            # Batch Normalization for standardization:
            nn.BatchNorm1d(hidden_dim),   

            # ReLU activation func:
            nn.ReLU(),
            
            # Ouput layer:            
            nn.Linear(hidden_dim, output_dim) 
        )

    # Forward pass:
    def forward(self, x):
        return self.net(x)
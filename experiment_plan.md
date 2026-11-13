
# Ahlam — BNN Experiment Plan

## Experiment 1: Weight Clipping
Compare BNN training with and without weight clipping to [-1, 1].

Measure:
- Test accuracy
- Training loss
- Weight values outside [-1, 1]

## Experiment 2: No-STE Control
Compare gradient behavior with and without the Straight-Through Estimator.

Measure:
- Gradient flow
- Training loss
- Test accuracy if training is possible

## Failure Diagnostics
Check for:
- Unstable loss
- Missing or zero gradients
- Weights outside [-1, 1]
- Incorrect predictions

## Shared Settings
Dataset: MNIST
Architecture: 784 -> 256 -> 10
Random seed: TBD
Optimizer: TBD
Learning rate: TBD
Batch size: TBD
Epochs: TBD

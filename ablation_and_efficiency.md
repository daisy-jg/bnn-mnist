# Full-Precision vs. BNN Ablation and Efficiency Profile

This report includes information about the quantitative ablation of full-precision MLP vs. BNN (STE + weight clipping) on MNIST, in addition to an efficiency profile (training time, memory, CPU utilization, parameter count). 

The raw data of the ablation results and memory and CPU logs can be found in:
- Accuracy and Training Time: `results/ablation_accuracy.csv`
- Memory and CPU: `logs/full_mem_cpu.log`, `logs/bnn_mem_cpu.log`


## Experimental Setup
Here are the training settings for this experiment:

| Setting | Value |
| :--- | :--- |
| Dataset | MNIST |
| Batch size | 100 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Number of epochs | 10 |
| Random seeds | 25, 30, 35 |
| Framework | PyTorch |


## Reproducibility
This experiment script initializes the following random seeds:
- Seed 25
- Seed 30
- Seed 35

To run all six configurations of two models and three seeds:
```bash
python3 ablation.py
```

To run a single configuration:
### Full-precision
```bash
python3 train_model.py --model full --epochs 10 --seed 25
python3 train_model.py --model full --epochs 10 --seed 30
python3 train_model.py --model full --epochs 10 --seed 35
```

### BNN
```bash
python3 train_model.py --model bnn --epochs 10 --seed 25
python3 train_model.py --model bnn --epochs 10 --seed 30
python3 train_model.py --model bnn --epochs 10 --seed 35
```


## Ablation: Full Precision vs. BNN

### Results table
| Model | Test Accuracy | Train Time (s) | Parameters | Seeds |
|---|---|---|---|---|
| Full precision | 97.90 ± 0.04 | 63.4 ± 2.1 | 204,042 | 25, 30, 35 |
| BNN (STE + clip) | 92.74 ± 0.36 | 70.7 ± 1.1 | 204,042 | 25, 30, 35 |
| **Paired gap (FP − BNN)** | **5.16 ± 0.48 pp** | — | — | — |

### Per-seed values
| Seed | Full precision | BNN | Paired gap (pp) |
|---|---|---|---|
| 25 | 97.95 | 92.23 | 5.72 |
| 30 | 97.86 | 92.95 | 4.91 |
| 35 | 97.89 | 93.04 | 4.85 |
| **Mean ± std** | **97.90 ± 0.04** | **92.74 ± 0.36** | **5.16 ± 0.48** |

### Discussion
The mean paired gap of 5.16 pp between the full-precision model and the BNN is greater than the value for each seed variation, which is observed as 0.04 pp for full-precision and 0.36 pp for BNN. This gap is noticeably larger than what was stated in the paper, where MNIST error rates are about 1.0–1.4% for BNNs (Hubara et al.). This discrepancy can be attributed to training differences, such as using fewer hidden units and training for only 10 epochs. 


## Efficiency Profile

### Results table
The table includes the type of hardware, input/batch size, and measurement tools used for this experiment.

| Measurement | Description |
| :--- | :--- |
| Computer | MacBook Pro |
| Operating system | macOS |
| Processing device | CPU |
| Framework | PyTorch |
| Memory measurement library | psutil |
| Timing library | Python time |
| CPU measurement | Python process CPU time and wall time |
| Batch size | 100 |
| Dataset | MNIST |
| Epochs | 10 |

### Memory and CPU Measurements
These measurements are from a single run using seed 25.

| Metric | Full-precision | BNN |
| :--- | :--- | :--- |
| Memory (between epochs) | 355.01 MB | 356.18 MB |
| Average CPU Utilization | 47.75% | 48.73% |
| Trainable Parameters | 204,042 | 204,042 |

The memory values reflect the amount of memory usage at the end of each epoch and do not represent the process's true peak memory. Since macOS has multi-cores usage, the average CPU utilization value is normalized by the total number of CPU cores on my computer. 

### Discussion
For both full-precision and BNN, the memory is almost the same, which is 355.01 MB for full-precision and 356.18 MB for BNN. BNN is approximately 11% slower on CPU. According to the paper, the speed depends on the XNOR bitwise kernels, but PyTorch CPU runtime uses floating-point matrix multiplications on binarized values. 


## Reference
Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). "Binarized Neural Networks." *Advances in Neural Information Processing Systems*, 29, 4107–4115.
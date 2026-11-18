# Lab 6 Report
## Task 1. YAML Configuration, MLflow and Checkpoints
The training process was made using a YAML configuration file
Implemented features:
- YAML configuration
- Random seed
- MLflow experiment logging
- Checkpoint saving
- Checkpoint loading

Final test accuracy: 0.9057

## Task 2. PyTorch Profiler
Profiler results showed that most execution time was spent in convolution and pooling operations.
- Convolution operations: approximately 58%
- Max Pooling operations: approximately 30%
- Other operations: approximately 12%

**Conclusion:**
Most computation time was spent in convolution layers.

## Task 3. Worker Study
| num_workers | Time (s) |
|------------|----------|
| 0 | 12.36 |
| 1 | 21.01 |
| 2 | 20.83 |
| 4 | 38.73 |
| 8 | 109.95 |

**Conclusion:**
The best results were with num_workers = 0.Increasing the number of workers increased execution time on this system

## Task 4. Batch Size Study
| Batch Size | Accuracy | Time (s) | Throughput (samples/s) |
|------------|----------|----------|----------|
| 32 | 0.9053 | 158.00 | 379.76 |
| 64 | 0.9034 | 1208.21 | 49.66 |
| 128 | 0.8947 | 211.31 | 283.95 |
| 256 | 0.8916 | 1206.81 | 49.72 |
| 512 | 0.8792 | 206.14 | 291.06 |

**Conclusion:**
The highest accuracy was achieved with batch size 32. Increasing batch size reduced accuracy

## Task 5. Optimizer and Scheduler Study
| Optimizer | Scheduler | Accuracy |
|------------|------------|------------|
| Adam | StepLR | 0.9038 |
| Adam | CosineAnnealingLR | 0.9023 |
| Adam | ReduceLROnPlateau | 0.8996 |
| SGD | StepLR | 0.9042 |
| SGD | CosineAnnealingLR | 0.9041 |
| SGD | ReduceLROnPlateau | 0.9088 |

**Conclusion:**
The best result was achieved using SGD with ReduceLROnPlateau

## Task 6. Gradient Accumulation
Configuration:
- Batch size = 64
- Accumulation steps = 4
- Effective batch size = 256
Conclusion:
Gradient accumulation successfully emulated a batch size of 256

## Task 7. Resume Training
Training was first runned for 3 epochs.
A checkpoint file *cnn.pt* was created and saved.
Training was then resumed from checkpoint and continued from epoch 4 to epoch 10
**Conclusion:**
Checkpoint restore worked successfully

## Task 8. Project Proposal
The project proposal is written in proposal.md


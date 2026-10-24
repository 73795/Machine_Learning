# Lab 2 — Neural Network from Scratch



## 1. Goal



The goal of this laboratory was to implement a neural network from scratch using NumPy. The network was used to classify handwritten digits from the `load_digits` dataset. The experiments studied how hidden-layer width and learning rate affect training loss and test accuracy.



## 2. Method



### Dataset and preprocessing



The experiment used the handwritten digits dataset from `sklearn.datasets.load_digits`. Each image contains 8 × 8 pixels, represented by 64 input features.



The pixel values were divided by 16 to normalize them. The labels were converted to one-hot encoded vectors with 10 classes, representing digits from 0 to 9.



The dataset was divided into training and testing sets using a stratified 70/30 split with `random_state=42`.



### Neural network architecture



The neural network was implemented using NumPy without a deep learning framework.



The architecture was:



- Input layer: 64 features

- First hidden layer: W neurons with ReLU activation

- Second hidden layer: W neurons with ReLU activation

- Output layer: 10 neurons with softmax activation



Two hidden-layer widths were tested: W = 16 and W = 64.



Three learning rates were tested: 0.001, 0.01, and 0.1. Each configuration was trained for 50 epochs.



The forward pass calculated predictions. Cross-entropy loss measured the difference between predictions and true labels. Backpropagation calculated gradients, and gradient descent updated the model parameters.



Gradient checking was used to compare analytical gradients with numerical gradients and help verify the implementation of backpropagation.



## 3. Results



The experiment compared six configurations. The results are presented in Table 1.



| Hidden width | Learning rate | Final training loss | Test accuracy |

|---:|---:|---:|---:|

| 16 | 0.001 | 2.30255 | 10.19% |

| 16 | 0.01 | 2.30248 | 10.19% |

| 16 | 0.1 | 1.59702 | 32.59% |

| 64 | 0.001 | 2.30253 | 10.19% |

| 64 | 0.01 | 2.30234 | 10.19% |

| 64 | 0.1 | 0.16840 | 93.89% |



The best result was achieved with a hidden-layer width of 64 and a learning rate of 0.1. This configuration reached a test accuracy of 93.89% and a final training loss of approximately 0.1684.



For learning rates of 0.001 and 0.01, the final loss remained close to 2.3026, and test accuracy was approximately 10.19%. This indicates that the model learned very little during these experiments.



The loss curves were saved in `results/loss_curves.png`, and the numerical results were saved in `results/training_results.csv`.



## 4. Discussion



The learning rate had a strong effect on the results. In these experiments, the learning rates of 0.001 and 0.01 were too small to produce effective learning within 50 epochs. The loss remained close to its initial value, and the accuracy was close to the level expected from random predictions across 10 classes.



With a learning rate of 0.1, the models performed better. The network with 16 neurons in each hidden layer achieved 32.59% test accuracy, while the network with 64 neurons achieved 93.89%.



This comparison shows that a wider network performed much better when combined with the learning rate of 0.1. However, the results do not prove that a wider network is always better. Performance depends on both architecture and training settings.



The loss curves help show how the models learned over time. The CSV file allows the numerical results of all six experiments to be compared.



## 5. Conclusions



1. A neural network for handwritten digit classification was implemented using NumPy.

2. The dataset was normalized and split into training and testing sets.

3. Six configurations were tested using two hidden-layer widths and three learning rates.

4. Learning rates of 0.001 and 0.01 did not produce effective learning within 50 epochs in these experiments.

5. The best configuration used a hidden-layer width of 64 and a learning rate of 0.1, achieving 93.89% test accuracy.

6. Both network architecture and learning rate are important for neural network performance.



## 6. Files



- `main.py` — neural network implementation and experiments.

- `results/training_results.csv` — numerical results for all configurations.

- `results/loss_curves.png` — training loss curves.



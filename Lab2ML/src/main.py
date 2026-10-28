import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


np.random.seed(42)

digits = load_digits()
X = digits.data.astype(np.float64) / 16.0
y = digits.target

Y = np.eye(10)[y]

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y, test_size=0.30, random_state=42, stratify=y
)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("One-hot labels shape:", Y_train.shape)


def initialise_parameters(width, seed=42):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0, 0.01, size=(64, width)),
        "b1": np.zeros((1, width)),
        "W2": rng.normal(0, 0.01, size=(width, width)),
        "b2": np.zeros((1, width)),
        "W3": rng.normal(0, 0.01, size=(width, 10)),
        "b3": np.zeros((1, 10)),
    }


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X_batch, params):
    Z1 = X_batch @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)

    Z3 = A2 @ params["W3"] + params["b3"]
    probabilities = softmax(Z3)

    cache = {
        "X": X_batch,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "probabilities": probabilities,
    }
    return probabilities, cache


def cross_entropy(probabilities, Y_batch):
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0)
    return -np.sum(Y_batch * np.log(probabilities)) / len(Y_batch)


def backward(Y_batch, params, cache):
    n = len(Y_batch)

    dZ3 = (cache["probabilities"] - Y_batch) / n
    dW3 = cache["A2"].T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2,
        "W3": dW3, "b3": db3,
    }


def gradient_check():
    params = initialise_parameters(width=16, seed=7)
    X_small = X_train[:8]
    Y_small = Y_train[:8]

    probabilities, cache = forward(X_small, params)
    analytical = backward(Y_small, params, cache)

    rng = np.random.default_rng(42)
    weight_names = ["W1", "W2", "W3"]
    checks = []

    for _ in range(10):
        name = rng.choice(weight_names)
        array = params[name]
        index = tuple(rng.integers(0, size) for size in array.shape)
        original = array[index]
        eps = 1e-5

        array[index] = original + eps
        plus_probs, _ = forward(X_small, params)
        loss_plus = cross_entropy(plus_probs, Y_small)

        array[index] = original - eps
        minus_probs, _ = forward(X_small, params)
        loss_minus = cross_entropy(minus_probs, Y_small)

        array[index] = original

        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytic = analytical[name][index]
        relative_error = abs(analytic - numerical) / max(
            1e-8, abs(analytic) + abs(numerical)
        )

        checks.append(relative_error)
        print(
            f"{name}{index}: analytical={analytic:.10e}, "
            f"numerical={numerical:.10e}, relative error={relative_error:.3e}"
        )

    passed = sum(error < 1e-6 for error in checks)
    print(f"Gradient check: {passed}/10 weights passed (need at least 9/10).")
    return passed


def predict(X_data, params):
    probabilities, _ = forward(X_data, params)
    return np.argmax(probabilities, axis=1)


def train(width, learning_rate, epochs=50, batch_size=32, seed=42):
    params = initialise_parameters(width, seed=seed)
    rng = np.random.default_rng(seed)
    losses = []

    for epoch in range(epochs):
        order = rng.permutation(len(X_train))
        X_shuffled = X_train[order]
        Y_shuffled = Y_train[order]

        for start in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            Y_batch = Y_shuffled[start:start + batch_size]

            probabilities, cache = forward(X_batch, params)
            gradients = backward(Y_batch, params, cache)

            for name in params:
                params[name] -= learning_rate * gradients[name]

        train_probs, _ = forward(X_train, params)
        epoch_loss = cross_entropy(train_probs, Y_train)
        losses.append(epoch_loss)

        print(
            f"W={width}, lr={learning_rate}, "
            f"epoch={epoch + 1:02d}/{epochs}, loss={epoch_loss:.6f}"
        )

    test_predictions = predict(X_test, params)
    accuracy = np.mean(test_predictions == y_test)
    return params, losses, accuracy


def measure_timing(params, runs=100):
    X_batch = X_train[:32]
    Y_batch = Y_train[:32]
    forward_times = []
    backward_times = []

    for _ in range(5):
        probs, cache = forward(X_batch, params)
        backward(Y_batch, params, cache)

    for _ in range(runs):
        start = time.perf_counter()
        probabilities, cache = forward(X_batch, params)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        backward(Y_batch, params, cache)
        backward_times.append(time.perf_counter() - start)

    forward_ms = np.median(forward_times) * 1000
    backward_ms = np.median(backward_times) * 1000

    print(f"Median forward time:  {forward_ms:.4f} ms")
    print(f"Median backward time: {backward_ms:.4f} ms")
    return forward_ms, backward_ms


os.makedirs("lab02/results", exist_ok=True)

params_check = initialise_parameters(width=16, seed=42)
initial_probs, initial_cache = forward(X_train[:32], params_check)

print("\nInitial probability output shape:", initial_probs.shape)
print("Row sums:", initial_probs.sum(axis=1)[:5])
print("Initial loss:", cross_entropy(initial_probs, Y_train[:32]))

assert initial_probs.shape == (32, 10)
assert np.allclose(initial_probs.sum(axis=1), 1.0, atol=1e-6)

print("\nRunning gradient check...")
gradient_check()

all_results = []
all_losses = {}
saved_params = {}

for width in [16, 64]:
    for learning_rate in [0.001, 0.01, 0.1]:
        print(f"\nStarting experiment: width={width}, lr={learning_rate}")
        params, losses, accuracy = train(
            width=width,
            learning_rate=learning_rate,
            epochs=50,
            batch_size=32,
            seed=42,
        )

        key = f"W={width}, lr={learning_rate}"
        all_losses[key] = losses
        saved_params[width] = params

        all_results.append({
            "width": width,
            "learning_rate": learning_rate,
            "final_train_loss": losses[-1],
            "test_accuracy": accuracy,
        })

        print(f"Final test accuracy: {accuracy:.4f}")

plt.figure(figsize=(10, 6))
for label, losses in all_losses.items():
    plt.plot(range(1, len(losses) + 1), losses, label=label)

plt.xlabel("Epoch")
plt.ylabel("Training cross-entropy loss")
plt.title("Loss curves for different widths and learning rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lab02/results/loss_curves.png", dpi=150)
plt.close()

with open("lab02/results/training_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["width", "learning_rate", "final_train_loss", "test_accuracy"],
    )
    writer.writeheader()
    writer.writerows(all_results)

print("\nTiming a trained network with width=16:")
measure_timing(saved_params[16], runs=100)

print("\nAll experiments finished.")
print("Saved: lab02/results/loss_curves.png")
print("Saved: lab02/results/training_results.csv")
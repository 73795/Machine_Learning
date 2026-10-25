
import os
import matplotlib.pyplot as plt


def create_plots(results):
    os.makedirs("results", exist_ok=True)

    names = list(results.keys())
    accuracy = [results[n]["test_accuracy"] for n in names]
    parameters = [results[n]["parameters"] for n in names]
    latency = [results[n]["latency_batch1_ms"] for n in names]

    plt.figure(figsize=(7, 5))
    plt.scatter(parameters, accuracy)

    for name in names:
        plt.annotate(
            name,
            (results[name]["parameters"],
             results[name]["test_accuracy"])
        )

    plt.xlabel("Trainable parameters")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs parameter count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_parameters.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(latency, accuracy)

    for name in names:
        plt.annotate(
            name,
            (results[name]["latency_batch1_ms"],
             results[name]["test_accuracy"])
        )

    plt.xlabel("Latency for batch size 1 (ms)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_latency.png")
    plt.close()

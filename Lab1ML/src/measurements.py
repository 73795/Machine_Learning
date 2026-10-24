import csv
import os
import statistics
import time
from pathlib import Path

import joblib
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

models = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

process = psutil.Process(os.getpid())
all_results = []

for name, create_model in models.items():
    print(f"\nMeasuring: {name}")

    model = create_model()
    model.fit(X_train, y_train)

    training_times = []

    for _ in range(5):
        model = create_model()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        training_times.append(end - start)

    training_median = statistics.median(training_times)

    sample = X_test[0].reshape(1, -1)
    inference_times = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        inference_times.append((end - start) * 1000)

    inference_median = statistics.median(inference_times)

    filename = name.lower().replace(" ", "_") + ".joblib"
    model_path = RESULTS_DIR / filename
    joblib.dump(model, model_path)

    model_size_bytes = model_path.stat().st_size
    model_size_kb = model_size_bytes / 1024

    memory_bytes = process.memory_info().rss
    memory_kb = memory_bytes / 1024

    accuracy = model.score(X_test, y_test)

    result = {
        "model": name,
        "accuracy": f"{accuracy:.4f}",
        "training_median_seconds": f"{training_median:.6f}",
        "inference_median_ms": f"{inference_median:.6f}",
        "model_size_bytes": model_size_bytes,
        "model_size_kb": f"{model_size_kb:.2f}",
        "process_memory_bytes": memory_bytes,
        "process_memory_kb": f"{memory_kb:.2f}",
    }

    all_results.append(result)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training median: {training_median:.6f} seconds")
    print(f"Inference median: {inference_median:.6f} ms")
    print(f"Model size: {model_size_kb:.2f} KB")
    print(f"Process memory: {memory_bytes} bytes ({memory_kb:.2f} KB)")


csv_path = RESULTS_DIR / "measurements.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nMeasurements saved to: {csv_path}")
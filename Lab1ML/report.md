\# Lab 1 — Baseline Models and System Cost



\## 1. Goal



The goal of this laboratory was to train baseline machine learning models and measure their system costs. The measurements included training time, single-sample inference latency, model size, and process memory usage. The results were used to evaluate the suitability of the models for Cloud, Edge, Mobile, and TinyML deployment.



\## 2. Method



The Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn was used.



The dataset was split into training and test sets using a 70/30 split. Stratification and `random\_state=42` were used to make the split reproducible.



Two classification models were trained:



\- Logistic Regression with `max\_iter=1000` and `random\_state=42`.

\- Random Forest with `n\_estimators=100` and `random\_state=42`.



Accuracy was measured on the test set.



For training time, one warm-up run was performed before five measured runs. The median training time was reported. Single-sample inference latency was measured over 100 repetitions, and its median was reported.



Both trained models were saved using joblib. Their saved file sizes were measured. Process memory usage was recorded using `psutil` and the process RSS value.



\## 3. Results



\### 3.1. Model accuracy



| Model | Test Accuracy |

|---|---:|

| Logistic Regression | 0.9415 |

| Random Forest | 0.9357 |



\### 3.2. System cost measurements



| Model | Median Training Time (s) | Median Inference Latency (ms) | Model Size (KB) | Process Memory (KB) |

|---|---:|---:|---:|---:|

| Logistic Regression | 0.851300 | 0.084200 | 1.03 | 107596.00 |

| Random Forest | 0.299318 | 4.476950 | 284.07 | 108896.00 |



The process memory values represent RSS at the time of measurement. They include Python and loaded libraries, not only the saved model.



\### 3.3. Deployment suitability



| Deployment | Memory Budget | Maximum Latency | Maximum Model Size | Logistic Regression | Random Forest |

|---|---|---|---|---|---|

| Cloud | At least 1 GB available | 100 ms | 500 MB | Suitable | Suitable |

| Edge | 256–1024 MB available | 50 ms | 50 MB | Suitable | Suitable |

| Mobile | 64–256 MB available | 20 ms | 10 MB | Suitable | Suitable |

| TinyML | Maximum 256 KB | 10 ms | 100 KB | Not suitable by measured process memory | Not suitable |



Both models meet the latency and saved model-size limits for Cloud, Edge, and Mobile deployment. The measured Python process memory is approximately 108 MB, so actual compatibility also depends on how much memory is available on the target device.



Neither model is suitable for TinyML based on the measured process memory. The Random Forest model also exceeds the TinyML model-size limit of 100 KB. Logistic Regression has a much smaller saved file, but its measured Python process memory is still above the TinyML memory budget.



\## 4. Conclusions



1\. Logistic Regression achieved slightly higher test accuracy than Random Forest: 0.9415 compared with 0.9357.



2\. Random Forest trained faster, with a median training time of 0.299318 seconds. Logistic Regression had much lower inference latency and a smaller saved model, making it more efficient when fast predictions and low storage requirements are important.



3\. Both models meet the stated latency and model-size limits for Cloud, Edge, and Mobile deployment, subject to the memory available on the target device. TinyML deployment is not suitable under the measured process-memory budget.




from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np
import hashlib
import joblib
import os

os.makedirs("results", exist_ok=True)

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy()

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

with open("results/validation_report.txt", "w") as file:
    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n")

violations = []

if not df["age"].between(17, 90).all():
    violations.append("Age outside range")

if not df["hours-per-week"].between(1, 99).all():
    violations.append("Hours outside range")

if not df["capital-gain"].between(0, 99999).all():
    violations.append("Capital gain outside range")

if not df["capital-loss"].between(0, 4356).all():
    violations.append("Capital loss outside range")

missing_count = (df == "?").sum().sum()

with open("results/validation_report.txt", "a") as file:

    file.write(f"\nMissing values: {missing_count}\n")

    if len(violations) == 0:
        file.write("No validation errors found\n")
    else:
        for item in violations:
            file.write(item + "\n")

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

train_df = pd.concat(
    [X_train, y_train],
    axis=1
)

before = len(train_df)

train_df = train_df.drop_duplicates()

after = len(train_df)

print("Duplicates removed:", before - after)

X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

numeric_cols_base = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

cat_cols_base = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_transformer_base = Pipeline([
    ("imputer",
     SimpleImputer(strategy="median"))
])

categorical_transformer_base = Pipeline([
    ("imputer",
     SimpleImputer(strategy="most_frequent")),
    ("encoder",
     OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_base = ColumnTransformer([
    ("num",
     numeric_transformer_base,
     numeric_cols_base),

    ("cat",
     categorical_transformer_base,
     cat_cols_base)
])

X_train_base = preprocessor_base.fit_transform(X_train)
X_test_base = preprocessor_base.transform(X_test)

model1 = LogisticRegression(max_iter=1000)

model1.fit(X_train_base, y_train)

pred1 = model1.predict(X_test_base)

acc1 = accuracy_score(y_test, pred1)

f11 = f1_score(
    y_test,
    pred1,
    pos_label=">50K"
)

def add_features(data):

    data = data.copy()

    data["capital_net"] = (
        data["capital-gain"]
        - data["capital-loss"]
    )

    data["hours_per_age"] = (
        data["hours-per-week"]
        / (data["age"] - 16)
    )

    return data


X_train_fe = add_features(X_train)
X_test_fe = add_features(X_test)

numeric_cols = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "capital_net",
    "hours_per_age"
]

for col in numeric_cols:

    low = X_train_fe[col].quantile(0.01)
    high = X_train_fe[col].quantile(0.99)

    X_train_fe[col] = X_train_fe[col].clip(low, high)
    X_test_fe[col] = X_test_fe[col].clip(low, high)

cat_cols = X_train_fe.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer",
     SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer",
     SimpleImputer(strategy="most_frequent")),

    ("encoder",
     OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num",
     numeric_transformer,
     numeric_cols),

    ("cat",
     categorical_transformer,
     cat_cols)
])

X_train_proc = preprocessor.fit_transform(
    X_train_fe
)

X_test_proc = preprocessor.transform(
    X_test_fe
)

joblib.dump(
    preprocessor,
    "results/preprocessor.joblib"
)

model2 = LogisticRegression(
    max_iter=1000
)

model2.fit(
    X_train_proc,
    y_train
)

pred2 = model2.predict(
    X_test_proc
)

from sklearn.metrics import confusion_matrix

print("\nCONFUSION MATRIX WITHOUT FEATURES")
print(confusion_matrix(
    y_test, pred1,
    labels=["<=50K", ">50K"]
))

print("\nCONFUSION MATRIX WITH FEATURES")
print(confusion_matrix(
    y_test, pred2,
    labels=["<=50K", ">50K"]
))

acc2 = accuracy_score(
    y_test,
    pred2
)

f12 = f1_score(
    y_test,
    pred2,
    pos_label=">50K"
)

print("\nWITHOUT FEATURES")
print("Accuracy:", acc1)
print("F1:", f11)

print("\nWITH FEATURES")
print("Accuracy:", acc2)
print("F1:", f12)

with open("results/metrics.txt", "w") as file:

    file.write("WITHOUT FEATURES\n")
    file.write(f"Accuracy: {acc1}\n")
    file.write(f"F1: {f11}\n\n")

    file.write("WITH FEATURES\n")
    file.write(f"Accuracy: {acc2}\n")
    file.write(f"F1: {f12}\n")

sample50 = df.sample(
    50,
    random_state=42
)

feature_cols = df.columns[:-1]

conflicts = (
    df.groupby(list(feature_cols))
    ["class"]
    .nunique()
)

conflict_count = (
    conflicts > 1
).sum()

print(
    "Conflicting duplicates:",
    conflict_count
)

with open(
    "results/validation_report.txt",
    "a"
) as file:

    file.write(
        f"\nConflicting duplicates: {conflict_count}\n"
    )

processed = pd.DataFrame(
    X_train_proc.toarray()
)

processed.to_csv(
    "results/processed_data.csv",
    index=False
)

with open(
    "results/processed_data.csv",
    "rb"
) as file:

    hash_value = hashlib.sha256(
        file.read()
    ).hexdigest()

with open(
    "results/dataset_hash.txt",
    "w"
) as file:

    file.write(hash_value)

print("\nSHA256 Hash:")
print(hash_value)

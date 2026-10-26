# Data Card: UCI Adult Dataset

## 1. Source

The Adult dataset is available from the UCI Machine Learning Repository and OpenML. It contains demographic and employment-related information used to predict income categories.

## 2. Size and Structure

The expected dataset size is 48,842 rows and 15 columns, including 14 input features and one target variable, `income`.

## 3. Intended Use

The dataset is used for educational purposes, including binary classification, data quality validation, preprocessing, feature engineering, and model evaluation.

## 4. License

The dataset's applicable license and usage terms must be checked in the official UCI and OpenML dataset documentation before redistribution.

## 5. Known Biases and Limitations

The dataset reflects historical demographic and employment patterns. It may contain sampling bias, historical bias, and unequal representation of demographic groups. Model predictions may differ across age, sex, race, and other groups. Income predictions should not be treated as objective judgments about individuals.

## 6. Privacy Concerns

The dataset contains demographic and employment-related information. Although it is commonly used for research and education, sensitive attributes and combinations of features may create fairness and privacy concerns. The data should not be used to make consequential decisions about real people without appropriate safeguards.

## 7. Data Quality

The dataset must be checked for missing values, invalid ranges, schema violations, duplicate records, and conflicting target labels. The original data should be preserved, and all preprocessing operations should be documented.

## 8. Reproducibility

The pipeline uses a fixed random seed, a stratified train/test split, training-only preprocessing, a saved preprocessing transformer, a YAML configuration, and a SHA256 hash of the processed dataset.

## 9. Evaluation

The Logistic Regression baseline is evaluated using accuracy and F1-score. Results are reported both with and without the engineered features.

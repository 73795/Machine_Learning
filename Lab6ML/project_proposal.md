# Project Proposal

## Project Title
Prediction of Unemployment Rate in Poland

## Problem
Predict rates of unemployment in Poland using data from demography and economy
to evaluate levels of unemployment in the future.

## Dataset
The dataset will be obtained from the Polish Central Statistical Office (GUS),
Bank Danych Lokalnych (BDL).

The dataset can include:
- Year
- Voivodeship
- Population
- Employment rate
- Number of registered enterprises
- Previous unemployment rate

## Model
MLP (Multi-Layer Perceptron) regression.

## Success Metric
The project will be evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Deployment Constraints
- Latency ≤ 50 ms
- Size ≤ 5 MB

## Planned Optimization
I will use quantization to make the model smaller and make it work better.
Then I will compare the old model with the new one.

## Expected Outcome
The goal is to make a machine learning model that can predict unemployment
rates in Poland with good accuracy.
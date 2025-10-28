# ml-models


install packages:
pip install pandas numpy scikit-learn matplotlib joblib


2️⃣ Australian House Price Prediction 🏡

Goal: Predict Melbourne house prices based on suburb, rooms, bathrooms, land size, and more.

Dataset: Melbourne Housing Market (Kaggle)

Model: HistGradientBoostingRegressor inside a full preprocessing Pipeline

Concepts Covered:

Handling missing values

OneHotEncoding categorical data

Model evaluation (MAE & RMSE)

Permutation importance for feature insights

Time-aware splits for realistic performance evaluation

Saving & loading models with joblib

📄 Files:

train_house_price_au.py — training pipeline

predict_one.py — single-sample prediction

melbourne_housing_sample.csv — example dataset

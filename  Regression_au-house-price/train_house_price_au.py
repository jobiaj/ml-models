# train_house_price_au.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import joblib
from math import sqrt


# --- 1) Load data ---
CSV_PATH = "melbourne_housing_sample.csv"  # e.g., Kaggle Melbourne Housing Market
df = pd.read_csv(CSV_PATH)

# Common columns in AU datasets (adjust if your CSV differs)
# Target
target_col = "Price"

# Try to parse Date if present for time-aware split
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Keep a reasonable subset of features (adjust to your CSV)
candidate_num = [
    c for c in ["Rooms","Bedroom2","Bathroom","Car","Landsize","BuildingArea",
                "YearBuilt","Distance","Propertycount","Postcode","Lattitude","Longtitude"]
    if c in df.columns
]
candidate_cat = [
    c for c in ["Suburb","Address","Type","Method","SellerG","CouncilArea","Regionname"]
    if c in df.columns
]

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# --- 2) Train/Test split (time-aware if Date exists) ---
if "Date" in df.columns and df["Date"].notna().sum() > 0:
    df = df.sort_values("Date")
    # Last 20% dates for test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df[candidate_num + candidate_cat]
y_train = train_df[target_col]
X_test  = test_df[candidate_num + candidate_cat]
y_test  = test_df[target_col]

# --- 3) Preprocess + Model pipeline ---
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median"))
    # For tree models, scaling isn’t required
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

pre = ColumnTransformer([
    ("num", num_pipe, candidate_num),
    ("cat", cat_pipe, candidate_cat)
])

# HistGradientBoosting is fast, strong baseline for tabular regression
model = HistGradientBoostingRegressor(
    max_depth=None,
    learning_rate=0.08,
    max_iter=500,
    l2_regularization=0.0,
    random_state=42,
    early_stopping=True
)

pipe = Pipeline([
    ("pre", pre),
    ("reg", model)
])

# --- 4) Cross-validate on training set ---
# If you have Date, use a time-series CV; else KFold via cross_val_score default.
if "Date" in train_df.columns and train_df["Date"].notna().sum() > 0:
    tscv = TimeSeriesSplit(n_splits=5)
    neg_mae = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error")
else:
    neg_mae = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

print(f"CV MAE:  {(-neg_mae).mean():.0f} ± {(-neg_mae).std():.0f}")
print(f"CV RMSE: {(-neg_rmse).mean():.0f} ± {(-neg_rmse).std():.0f}")

# --- 5) Fit and evaluate on hold-out test set ---
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = sqrt(mean_squared_error(y_test, pred))
print(f"TEST MAE:  {mae:.0f}")
print(f"TEST RMSE: {rmse:.0f}")


# --- Permutation importance per RAW input column ---
test_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
y_sample   = y_test.loc[test_sample.index]

r = permutation_importance(
    pipe,                 # note: the whole pipeline
    test_sample, y_sample,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42
)

feature_names = list(test_sample.columns)  # raw column names you permuted
importances = pd.DataFrame({
    "feature": feature_names,
    "importance": r.importances_mean
}).sort_values("importance", ascending=False)

print("\nTop features by permutation importance (raw columns):")
print(importances.head(15).to_string(index=False))

# --- 7) Save model for serving/inference ---
joblib.dump(pipe, "au_house_price_model.joblib")
print("\nSaved model to au_house_price_model.joblib")

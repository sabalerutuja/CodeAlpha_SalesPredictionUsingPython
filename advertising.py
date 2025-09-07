# sales_prediction.py
# End-to-end Sales Prediction using Advertising.csv
# - Cleans & prepares data
# - Feature engineering (logs, interactions, lags if date present)
# - Trains Ridge (interpretable) and Random Forest (nonlinear)
# - Evaluates with RMSE & R^2
# - What-if analysis: effect of increasing ad spend by channel
# - Optional time-aware split if a Date column exists

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------
# 1) Load & clean
# ------------------
csv_path = "Advertising.csv"  # keep file in the same folder
if not os.path.exists(csv_path):
    # fallback for environments where it's placed elsewhere
    csv_path = "/mnt/data/Advertising.csv"

df = pd.read_csv(csv_path)

# Standardize column names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("-", "_")
)

# Common schemas:
# - ISLR dataset: tv, radio, newspaper, sales
# - Extended datasets might include: date, platform, segment, region, etc.
# Try to infer target & features
possible_target_names = ["sales", "target", "revenue", "units"]
target_col = next((c for c in possible_target_names if c in df.columns), None)
if target_col is None:
    raise ValueError(
        f"Could not find a target column. Expected one of {possible_target_names} in CSV."
    )

# Try to detect a date column
date_col = None
for cand in ["date", "week", "month", "period"]:
    if cand in df.columns:
        date_col = cand
        break
if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Basic cleaning: drop duplicate rows, drop rows with missing target
df = df.drop_duplicates()
df = df.dropna(subset=[target_col])

# -------------------------------
# 2) Feature engineering (flex)
# -------------------------------
# Identify spend-like numeric columns (heuristic)
numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_raw = [c for c in numeric_cols_raw if c != target_col]

categorical_cols_raw = df.select_dtypes(include=["object", "category"]).columns.tolist()
# If date exists, remove it from categorical list
if date_col in categorical_cols_raw:
    categorical_cols_raw.remove(date_col)

# Log transforms for diminishing returns (tv, radio, newspaper, search, social, etc.)
def looks_like_spend(col):
    spend_keywords = ["tv", "radio", "newspaper", "social", "search", "digital", "display", "facebook", "google", "youtube"]
    return any(k in col for k in spend_keywords)

spend_cols = [c for c in numeric_cols_raw if looks_like_spend(c)]
for c in spend_cols:
    df[f"log1p_{c}"] = np.log1p(df[c])

# Simple two-way interactions across ad channels (if present)
if len(spend_cols) >= 2:
    for i in range(len(spend_cols)):
        for j in range(i + 1, len(spend_cols)):
            a, b = spend_cols[i], spend_cols[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]

# Lag features if date is present (sorted time-aware)
if date_col is not None:
    df = df.sort_values(by=date_col).reset_index(drop=True)
    for c in spend_cols:
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_ma3"] = df[c].rolling(3).mean()
    # Drop initial NA from lags
    df = df.dropna().reset_index(drop=True)

# Final feature list (exclude target & date)
feature_cols = [c for c in df.columns if c not in [target_col, date_col]]

# Split features by type
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

# --------------------------------------
# 3) Train/Test split (time-aware if possible)
# --------------------------------------
if date_col is not None:
    # Last 20% as test (hold-out by time)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train, y_train = train_df[feature_cols], train_df[target_col].values
X_test, y_test = test_df[feature_cols], test_df[target_col].values

# --------------------------------------
# 4) Preprocess & models
# --------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

ridge = Pipeline(steps=[("prep", preprocess), ("model", Ridge(alpha=1.0, random_state=42))])
rf = Pipeline(steps=[("prep", preprocess), ("model", RandomForestRegressor(n_estimators=500, random_state=42))])

ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)

pred_ridge = ridge.predict(X_test)
pred_rf = rf.predict(X_test)

def report_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # manual RMSE
    r2 = r2_score(y_true, y_pred)
    print(f"{name:8s} -> RMSE: {rmse:.3f} | R^2: {r2:.3f}")



print("\nModel performance on test set")
report_metrics("Ridge", y_test, pred_ridge)
report_metrics("RF", y_test, pred_rf)

# --------------------------------------
# 5) Plots (Actual vs Predicted) — Ridge
# --------------------------------------
plt.figure(figsize=(10, 5))
if date_col is not None:
    plt.plot(test_df[date_col], y_test, label="Actual")
    plt.plot(test_df[date_col], pred_ridge, label="Predicted (Ridge)")
    plt.xlabel("Date")
else:
    plt.plot(y_test, label="Actual")
    plt.plot(pred_ridge, label="Predicted (Ridge)")
    plt.xlabel("Sample")
plt.ylabel(target_col.title())
plt.title("Actual vs Predicted (Ridge)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
mn, mx = min(y_test.min(), pred_ridge.min()), max(y_test.max(), pred_ridge.max())
plt.scatter(y_test, pred_ridge)
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual")
plt.ylabel("Predicted (Ridge)")
plt.title("Actual vs Predicted Scatter")
plt.tight_layout()
plt.show()

# --------------------------------------
# 6) Interpretability: standardized coefficients (Ridge)
# --------------------------------------
# Recover names after preprocessing
ohe = ridge.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
feature_names = num_cols + cat_feature_names
coefs = ridge.named_steps["model"].coef_

coef_table = pd.DataFrame({"feature": feature_names, "coef_standardized": coefs})
coef_table = coef_table.sort_values("coef_standardized", ascending=False)
print("\nTop positive standardized drivers:\n", coef_table.head(10).to_string(index=False))
print("\nTop negative standardized drivers:\n", coef_table.tail(10).to_string(index=False))

# --------------------------------------
# 7) What-if analysis: ad spend impact
#    Increase each spend channel by {+10, +20, +30}% and compute avg predicted lift
# --------------------------------------
spend_like = spend_cols[:]  # re-use detected spend columns
if not spend_like:
    # Fallback to common ISLR names if auto-detection failed
    for c in ["tv", "radio", "newspaper"]:
        if c in df.columns:
            spend_like.append(c)

def what_if(table: pd.DataFrame, model: Pipeline, pct_list=(10, 20, 30)):
    rows = []
    base_pred = model.predict(table[feature_cols]).mean()
    for ch in spend_like:
        for pct in pct_list:
            temp = table.copy()
            temp[ch] = temp[ch] * (1 + pct / 100.0)
            # keep lags/MAs unchanged for short-horizon scenario; adequate for directional insight
            new_pred = model.predict(temp[feature_cols]).mean()
            rows.append({"channel": ch, "increase_%": pct, "avg_sales_lift": round(new_pred - base_pred, 3)})
    return pd.DataFrame(rows).sort_values(["channel", "increase_%"])

# Use last N rows (or test set) for scenario analysis
scenario_df = test_df.copy()
lift_df = what_if(scenario_df, ridge, pct_list=(10, 20, 30))
print("\nWhat-if: average predicted lift when increasing spend")
print(lift_df.to_string(index=False))

"""
=============================================================================
  Tesla Deliveries ML Pipeline (2015–2025)
  Author : Akarsh Kumar
  Course : B.Tech CSE (Data Science & Analytics), DIT University
  Dataset: Tesla Deliveries Dataset 2015-2025 (2,640 records × 12 features)
=============================================================================
  Pipeline Stages
  ───────────────
  0.  Imports & Configuration
  1.  Data Loading & Initial Inspection
  2.  Exploratory Data Analysis (EDA)
  3.  Preprocessing & Feature Engineering
  4.  Regression Modeling (Price Prediction)
      4a. Baseline Models
      4b. LightGBM / XGBoost
      4c. Hyperparameter Tuning (GridSearchCV)
      4d. Model Comparison & Feature Importance
  5.  Time-Series Forecasting (Deliveries → 2026-2027)
      5a. SARIMA
      5b. Prophet
  6.  Results Summary
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import lightgbm as lgb
import xgboost as xgb

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

# ── aesthetic defaults ────────────────────────────────────────────────────────
PALETTE  = ["#E31937", "#1A1A2E", "#2E86AB", "#F0A500", "#4CAF50", "#9C27B0"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans",
})

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "/mnt/user-data/uploads/tesla_deliveries_dataset_2015_2025.csv"

def save(fig, name):
    """Save figure to output directory."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {path}")

def section(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & INITIAL INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
section("1. DATA LOADING & INITIAL INSPECTION")

df = pd.read_csv(DATA_PATH)
print(f"Shape           : {df.shape}")
print(f"Columns         : {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive statistics:\n{df.describe()}")

CATS   = ["Region", "Model", "Source_Type"]
NUMS   = ["Estimated_Deliveries", "Production_Units", "Avg_Price_USD",
          "Battery_Capacity_kWh", "Range_km", "CO2_Saved_tons", "Charging_Stations"]
TARGET = "Avg_Price_USD"

print("\nUnique values per categorical column:")
for c in CATS:
    print(f"  {c}: {sorted(df[c].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("2. EDA")

# ── 2-A  Distribution of the target ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("2-A  Target Distribution: Avg_Price_USD", fontweight="bold")

axes[0].hist(df[TARGET], bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.4)
axes[0].set_title("Histogram")
axes[0].set_xlabel("Avg Price (USD)")
axes[0].set_ylabel("Frequency")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

axes[1].boxplot(df[TARGET], vert=True, patch_artist=True,
                boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
axes[1].set_title("Box Plot")
axes[1].set_ylabel("Avg Price (USD)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))

plt.tight_layout()
save(fig, "01_target_distribution.png")

# ── 2-B  Deliveries over time ────────────────────────────────────────────────
ts = df.groupby(["Year","Month"])["Estimated_Deliveries"].sum().reset_index()
ts["Date"] = pd.to_datetime(ts.assign(Day=1)[["Year","Month","Day"]])
ts.sort_values("Date", inplace=True)

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("2-B  Tesla Global Deliveries Over Time (2015–2025)", fontweight="bold")
ax.fill_between(ts["Date"], ts["Estimated_Deliveries"], alpha=0.25, color=PALETTE[0])
ax.plot(ts["Date"], ts["Estimated_Deliveries"], color=PALETTE[0], linewidth=1.6)
ax.set_xlabel("Date")
ax.set_ylabel("Total Deliveries")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1000:.0f}K"))
plt.tight_layout()
save(fig, "02_deliveries_over_time.png")

# ── 2-C  Deliveries by Region ────────────────────────────────────────────────
reg = df.groupby("Region")["Estimated_Deliveries"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("2-C  Total Deliveries by Region", fontweight="bold")
bars = ax.bar(reg.index, reg.values, color=PALETTE[:len(reg)], edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f"{bar.get_height()/1e6:.2f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Estimated Deliveries")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e6:.1f}M"))
plt.tight_layout()
save(fig, "03_deliveries_by_region.png")

# ── 2-D  Deliveries by Model ─────────────────────────────────────────────────
mod = df.groupby("Model")["Estimated_Deliveries"].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("2-D  Total Deliveries by Model", fontweight="bold")
wedges, texts, autotexts = ax.pie(mod.values, labels=mod.index, autopct="%1.1f%%",
                                   colors=PALETTE[:len(mod)], startangle=140,
                                   pctdistance=0.82)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Market Share by Model")
plt.tight_layout()
save(fig, "04_deliveries_by_model.png")

# ── 2-E  Avg Price by Model (boxplot) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("2-E  Price Distribution by Model", fontweight="bold")
order = df.groupby("Model")["Avg_Price_USD"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Model", y="Avg_Price_USD", order=order,
            palette=PALETTE[:5], ax=ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))
ax.set_xlabel("Model")
ax.set_ylabel("Avg Price (USD)")
plt.tight_layout()
save(fig, "05_price_by_model.png")

# ── 2-F  Correlation heatmap ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("2-F  Correlation Heatmap (Numerical Features)", fontweight="bold")
corr = df[NUMS + ["Year","Month"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
plt.tight_layout()
save(fig, "06_correlation_heatmap.png")

# ── 2-G  YoY Price trend by Model ────────────────────────────────────────────
yoy = df.groupby(["Year","Model"])["Avg_Price_USD"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("2-G  Average Price Trend by Model (2015–2025)", fontweight="bold")
for i, model in enumerate(df["Model"].unique()):
    sub = yoy[yoy["Model"] == model]
    ax.plot(sub["Year"], sub["Avg_Price_USD"], marker="o", linewidth=2,
            label=model, color=PALETTE[i])
ax.set_xlabel("Year")
ax.set_ylabel("Avg Price (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))
ax.legend(title="Model", framealpha=0.9)
plt.tight_layout()
save(fig, "07_price_trend_by_model.png")

# ── 2-H  CO2 Saved vs Deliveries scatter ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("2-H  CO₂ Saved vs. Estimated Deliveries", fontweight="bold")
for i, model in enumerate(df["Model"].unique()):
    sub = df[df["Model"] == model]
    ax.scatter(sub["Estimated_Deliveries"], sub["CO2_Saved_tons"],
               alpha=0.45, s=25, label=model, color=PALETTE[i])
ax.set_xlabel("Estimated Deliveries")
ax.set_ylabel("CO₂ Saved (tons)")
ax.legend(title="Model")
plt.tight_layout()
save(fig, "08_co2_vs_deliveries.png")

print("  ✓ All EDA charts saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
section("3. PREPROCESSING & FEATURE ENGINEERING")

df_ml = df.copy()

# ── 3-A  Encode categoricals ──────────────────────────────────────────────────
le_region = LabelEncoder()
le_model  = LabelEncoder()
le_source = LabelEncoder()

df_ml["Region_enc"]      = le_region.fit_transform(df_ml["Region"])
df_ml["Model_enc"]       = le_model.fit_transform(df_ml["Model"])
df_ml["Source_Type_enc"] = le_source.fit_transform(df_ml["Source_Type"])

# ── 3-B  Time features ────────────────────────────────────────────────────────
df_ml["Quarter"]         = ((df_ml["Month"] - 1) // 3 + 1).astype(int)
df_ml["Month_sin"]       = np.sin(2 * np.pi * df_ml["Month"] / 12)
df_ml["Month_cos"]       = np.cos(2 * np.pi * df_ml["Month"] / 12)
df_ml["Years_since_2015"] = df_ml["Year"] - 2015

# ── 3-C  Derived features ─────────────────────────────────────────────────────
df_ml["Delivery_Ratio"]   = (df_ml["Estimated_Deliveries"] /
                              df_ml["Production_Units"].replace(0, np.nan)).fillna(0)
df_ml["CO2_per_Delivery"] = (df_ml["CO2_Saved_tons"] /
                              df_ml["Estimated_Deliveries"].replace(0, np.nan)).fillna(0)
df_ml["Price_per_kWh"]    = df_ml["Avg_Price_USD"] / df_ml["Battery_Capacity_kWh"]
df_ml["Station_Density"]  = (df_ml["Charging_Stations"] /
                              df_ml["Estimated_Deliveries"].replace(0, np.nan)).fillna(0)
df_ml["Range_per_kWh"]    = df_ml["Range_km"] / df_ml["Battery_Capacity_kWh"]

# ── 3-D  Feature set for regression ─────────────────────────────────────────
FEATURES = [
    "Year", "Month", "Quarter", "Month_sin", "Month_cos", "Years_since_2015",
    "Region_enc", "Model_enc", "Source_Type_enc",
    "Estimated_Deliveries", "Production_Units",
    "Battery_Capacity_kWh", "Range_km", "CO2_Saved_tons", "Charging_Stations",
    "Delivery_Ratio", "CO2_per_Delivery", "Price_per_kWh",
    "Station_Density", "Range_per_kWh",
]

X = df_ml[FEATURES]
y = df_ml[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Features used: {len(FEATURES)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. REGRESSION MODELING
# ─────────────────────────────────────────────────────────────────────────────
section("4. REGRESSION MODELING  →  Target: Avg_Price_USD")

def evaluate(name, model, X_tr, X_te, y_tr, y_te, scaled=False):
    """Fit, predict, print and return metrics dict."""
    Xtr = X_tr if not scaled else scaler.transform(X_tr)
    Xte = X_te if not scaled else scaler.transform(X_te)
    model.fit(Xtr, y_tr)
    pred = model.predict(Xte)
    mae  = mean_absolute_error(y_te, pred)
    rmse = mean_squared_error(y_te, pred) ** 0.5
    r2   = r2_score(y_te, pred)
    cv   = cross_val_score(model, Xtr, y_tr, cv=5,
                           scoring="r2", n_jobs=-1).mean()
    print(f"  {name:<35} MAE=${mae:>9,.0f}  RMSE=${rmse:>9,.0f}  R²={r2:.4f}  CV-R²={cv:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv,
            "predictions": pred}

results = {}

# ── 4-A  Baseline models ─────────────────────────────────────────────────────
print("\n── 4-A  Baseline Models ──")
results["Linear Regression"] = evaluate(
    "Linear Regression", LinearRegression(), X_train_sc, X_test_sc, y_train, y_test)

results["Ridge (α=1)"] = evaluate(
    "Ridge (α=1)", Ridge(alpha=1.0), X_train_sc, X_test_sc, y_train, y_test)

results["Lasso (α=1)"] = evaluate(
    "Lasso (α=1)", Lasso(alpha=1.0, max_iter=5000), X_train_sc, X_test_sc, y_train, y_test)

results["Random Forest"] = evaluate(
    "Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test)

results["Gradient Boosting"] = evaluate(
    "Gradient Boosting", GradientBoostingRegressor(n_estimators=200, random_state=42),
    X_train, X_test, y_train, y_test)

# ── 4-B  LightGBM & XGBoost ─────────────────────────────────────────────────
print("\n── 4-B  LightGBM / XGBoost ──")
lgb_base = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                               num_leaves=63, random_state=42, verbose=-1)
results["LightGBM (base)"] = evaluate(
    "LightGBM (base)", lgb_base, X_train, X_test, y_train, y_test)

xgb_base = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=6, random_state=42, verbosity=0)
results["XGBoost (base)"] = evaluate(
    "XGBoost (base)", xgb_base, X_train, X_test, y_train, y_test)

# ── 4-C  Hyperparameter Tuning ───────────────────────────────────────────────
section("4-C  Hyperparameter Tuning  (GridSearchCV  cv=5)")

lgb_param_grid = {
    "n_estimators"  : [200, 400],
    "learning_rate" : [0.03, 0.07],
    "num_leaves"    : [31, 63, 127],
    "min_child_samples": [20, 50],
}

print("  Tuning LightGBM … (this may take ~1 min)")
gs_lgb = GridSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1),
    lgb_param_grid,
    cv=5, scoring="r2", n_jobs=-1, verbose=0
)
gs_lgb.fit(X_train, y_train)
print(f"  Best params : {gs_lgb.best_params_}")
print(f"  Best CV-R²  : {gs_lgb.best_score_:.4f}")

results["LightGBM (tuned)"] = evaluate(
    "LightGBM (tuned)", gs_lgb.best_estimator_, X_train, X_test, y_train, y_test)

xgb_param_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.07],
    "max_depth"   : [4, 6],
    "subsample"   : [0.8, 1.0],
}

print("\n  Tuning XGBoost … (this may take ~1 min)")
gs_xgb = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    xgb_param_grid,
    cv=5, scoring="r2", n_jobs=-1, verbose=0
)
gs_xgb.fit(X_train, y_train)
print(f"  Best params : {gs_xgb.best_params_}")
print(f"  Best CV-R²  : {gs_xgb.best_score_:.4f}")

results["XGBoost (tuned)"] = evaluate(
    "XGBoost (tuned)", gs_xgb.best_estimator_, X_train, X_test, y_train, y_test)

# ── 4-D  Model Comparison Charts ─────────────────────────────────────────────
section("4-D  Model Comparison Visuals")

summary = pd.DataFrame([
    {"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"], "CV_R2": v["CV_R2"]}
    for k, v in results.items()
]).sort_values("R2", ascending=False).reset_index(drop=True)

print(f"\n{summary.to_string(index=False)}")

# 4-D-i  R² comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("4-D-i  Model R² Comparison (Test Set)", fontweight="bold")
colors = [PALETTE[0] if i == 0 else PALETTE[2] for i in range(len(summary))]
bars = ax.barh(summary["Model"], summary["R2"], color=colors, edgecolor="white")
for bar, val in zip(bars, summary["R2"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("R² Score")
ax.set_xlim(0, 1.05)
ax.invert_yaxis()
plt.tight_layout()
save(fig, "09_model_r2_comparison.png")

# 4-D-ii  RMSE comparison
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("4-D-ii  Model RMSE Comparison (Test Set)", fontweight="bold")
bars = ax.barh(summary["Model"], summary["RMSE"], color=PALETTE[1], edgecolor="white", alpha=0.8)
for bar, val in zip(bars, summary["RMSE"]):
    ax.text(val + 50, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9)
ax.set_xlabel("RMSE (USD)")
ax.invert_yaxis()
plt.tight_layout()
save(fig, "10_model_rmse_comparison.png")

# 4-D-iii  Best model: Actual vs Predicted
best_name = summary.iloc[0]["Model"]
best_pred = results[best_name]["predictions"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"4-D-iii  Best Model: {best_name}  —  Actual vs Predicted", fontweight="bold")

axes[0].scatter(y_test, best_pred, alpha=0.35, s=18, color=PALETTE[0], label="Predictions")
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
axes[0].plot(lims, lims, "k--", linewidth=1.5, label="Perfect Fit")
axes[0].set_xlabel("Actual Price (USD)")
axes[0].set_ylabel("Predicted Price (USD)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))
axes[0].legend()

residuals = y_test.values - best_pred
axes[1].scatter(best_pred, residuals, alpha=0.35, s=18, color=PALETTE[3])
axes[1].axhline(0, color="black", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Predicted Price (USD)")
axes[1].set_ylabel("Residual (USD)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}K"))
axes[1].set_title("Residual Plot")

plt.tight_layout()
save(fig, "11_actual_vs_predicted.png")

# 4-D-iv  Feature Importance (best tree model)
best_model_obj = gs_lgb.best_estimator_ if "LightGBM" in best_name else gs_xgb.best_estimator_

fi = pd.Series(best_model_obj.feature_importances_, index=FEATURES).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle(f"4-D-iv  Feature Importance ({best_name})", fontweight="bold")
fi.head(15).plot.barh(ax=ax, color=PALETTE[0], edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
plt.tight_layout()
save(fig, "12_feature_importance.png")

print(f"\n  Top 5 features:\n{fi.head()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TIME-SERIES FORECASTING  →  Global Deliveries
# ─────────────────────────────────────────────────────────────────────────────
section("5. TIME-SERIES FORECASTING  →  Global Monthly Deliveries")

# Build monthly global delivery series
ts_df = (df.groupby(["Year","Month"])["Estimated_Deliveries"]
           .sum().reset_index())
ts_df["Date"] = pd.to_datetime(
    ts_df["Year"].astype(str) + "-" + ts_df["Month"].astype(str).str.zfill(2) + "-01"
)
ts_df.sort_values("Date", inplace=True)
ts_df.set_index("Date", inplace=True)

y_ts = ts_df["Estimated_Deliveries"].astype(float)

# ── 5-A  Stationarity check ───────────────────────────────────────────────────
adf_result = adfuller(y_ts)
print(f"  ADF Statistic : {adf_result[0]:.4f}")
print(f"  p-value       : {adf_result[1]:.4f}")
print(f"  Stationary    : {'Yes' if adf_result[1] < 0.05 else 'No (differencing needed)'}")

# Differenced for SARIMA
y_diff = y_ts.diff(12).dropna()

# ── 5-B  SARIMA ───────────────────────────────────────────────────────────────
print("\n── 5-B  SARIMA(1,1,1)(1,1,1,12) ──")
try:
    sarima = SARIMAX(y_ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                     enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=False)
    print(sarima_fit.summary().tables[0])

    FORECAST_STEPS = 24   # 2 years
    sarima_fc = sarima_fit.get_forecast(steps=FORECAST_STEPS)
    fc_mean  = sarima_fc.predicted_mean
    fc_ci    = sarima_fc.conf_int()
    fc_index = pd.date_range(y_ts.index[-1] + pd.DateOffset(months=1),
                             periods=FORECAST_STEPS, freq="MS")
    fc_mean.index = fc_index
    fc_ci.index   = fc_index

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("5-B  SARIMA Forecast — Monthly Deliveries (2026–2027)", fontweight="bold")
    ax.plot(y_ts, label="Historical", color=PALETTE[0], linewidth=1.6)
    ax.plot(fc_mean, label="SARIMA Forecast", color=PALETTE[2],
            linewidth=2, linestyle="--", marker="o", markersize=3)
    ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1],
                    alpha=0.25, color=PALETTE[2], label="95% CI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deliveries")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1000:.0f}K"))
    ax.legend()
    plt.tight_layout()
    save(fig, "13_sarima_forecast.png")

    sarima_ok = True
    print(f"  SARIMA AIC: {sarima_fit.aic:.2f}  |  BIC: {sarima_fit.bic:.2f}")
except Exception as e:
    print(f"  ⚠ SARIMA failed: {e}")
    sarima_ok = False

# ── 5-C  Prophet ─────────────────────────────────────────────────────────────
print("\n── 5-C  Facebook Prophet ──")
prophet_df = y_ts.reset_index().rename(columns={"Date":"ds","Estimated_Deliveries":"y"})
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
            daily_seasonality=False, changepoint_prior_scale=0.3)
m.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=24, freq="MS")
forecast_p = m.predict(future)

fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("5-C  Prophet Forecast — Monthly Deliveries (2026–2027)", fontweight="bold")
ax.plot(prophet_df["ds"], prophet_df["y"],
        label="Historical", color=PALETTE[0], linewidth=1.6)
fc_only = forecast_p[forecast_p["ds"] > prophet_df["ds"].max()]
ax.plot(forecast_p["ds"], forecast_p["yhat"],
        label="Prophet Forecast", color=PALETTE[4], linewidth=2, linestyle="--")
ax.fill_between(forecast_p["ds"],
                forecast_p["yhat_lower"], forecast_p["yhat_upper"],
                alpha=0.2, color=PALETTE[4], label="Uncertainty Interval")
ax.axvline(prophet_df["ds"].max(), color="gray", linestyle=":", linewidth=1.2,
           label="Forecast Start")
ax.set_xlabel("Date")
ax.set_ylabel("Deliveries")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1000:.0f}K"))
ax.legend()
plt.tight_layout()
save(fig, "14_prophet_forecast.png")

# Prophet components
fig2 = m.plot_components(forecast_p)
fig2.suptitle("5-C  Prophet Forecast Components", fontweight="bold", y=1.01)
save(fig2, "15_prophet_components.png")

# ── 5-D  Model Comparison — last 12 months hold-out ─────────────────────────
holdout_start = y_ts.index[-12]
y_train_ts = y_ts[:holdout_start]
y_test_ts  = y_ts[holdout_start:]

if sarima_ok:
    sarima2 = SARIMAX(y_train_ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                      enforce_stationarity=False, enforce_invertibility=False)
    sarima2_fit = sarima2.fit(disp=False)
    sarima_hf   = sarima2_fit.get_forecast(steps=12).predicted_mean

prophet_df2 = y_train_ts.reset_index().rename(columns={"Date":"ds","Estimated_Deliveries":"y"})
prophet_df2["ds"] = pd.to_datetime(prophet_df2["ds"])
m2 = Prophet(yearly_seasonality=True, weekly_seasonality=False,
             daily_seasonality=False, changepoint_prior_scale=0.3)
m2.fit(prophet_df2)
future2 = m2.make_future_dataframe(periods=12, freq="MS")
fc2 = m2.predict(future2)
prophet_hf = fc2.tail(12)["yhat"].values

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("5-D  Forecasting Methods — 12-Month Hold-Out Comparison", fontweight="bold")
ax.plot(y_test_ts.index, y_test_ts.values, label="Actual", linewidth=2, color=PALETTE[0], marker="o", markersize=4)
if sarima_ok:
    ax.plot(y_test_ts.index, sarima_hf.values, label="SARIMA", linewidth=2,
            color=PALETTE[2], linestyle="--", marker="s", markersize=4)
ax.plot(y_test_ts.index, prophet_hf, label="Prophet", linewidth=2,
        color=PALETTE[4], linestyle="-.", marker="^", markersize=4)
ax.set_xlabel("Date")
ax.set_ylabel("Deliveries")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1000:.0f}K"))
ax.legend()
plt.tight_layout()
save(fig, "16_ts_model_comparison.png")

# TS metrics
print("\n  Time-Series Hold-Out Metrics (last 12 months)")
if sarima_ok:
    s_mae  = mean_absolute_error(y_test_ts, sarima_hf)
    s_rmse = mean_squared_error(y_test_ts, sarima_hf) ** 0.5
    print(f"  SARIMA  → MAE={s_mae:,.0f}  RMSE={s_rmse:,.0f}")

p_mae  = mean_absolute_error(y_test_ts, prophet_hf)
p_rmse = mean_squared_error(y_test_ts, prophet_hf) ** 0.5
print(f"  Prophet → MAE={p_mae:,.0f}  RMSE={p_rmse:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("6. RESULTS SUMMARY")

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │            TESLA DELIVERIES ML PIPELINE — SUMMARY               │
  ├─────────────────────────────────────────────────────────────────┤
  │  Dataset      : 2,640 records × 12 features  (2015–2025)        │
  │  Target (reg) : Avg_Price_USD                                   │
  │  Target (ts)  : Monthly Estimated_Deliveries                    │
  │                                                                 │
  │  Best Regression Model : {best_name:<30}        │
  │    R²   = {summary.iloc[0]['R2']:.4f}                                          │
  │    RMSE = ${summary.iloc[0]['RMSE']:>9,.0f}                                     │
  │    MAE  = ${summary.iloc[0]['MAE']:>9,.0f}                                     │
  │                                                                 │
  │  Features Engineered : {len(FEATURES)} (incl. cyclic, ratio, derived)       │
  │  Tuning              : GridSearchCV  (cv=5)                     │
  │  Time-Series Models  : SARIMA + Prophet (24-month forecast)     │
  ├─────────────────────────────────────────────────────────────────┤
  │  Charts saved → /mnt/user-data/outputs/                         │
  │    01_target_distribution.png                                   │
  │    02_deliveries_over_time.png                                  │
  │    03_deliveries_by_region.png                                  │
  │    04_deliveries_by_model.png                                   │
  │    05_price_by_model.png                                        │
  │    06_correlation_heatmap.png                                   │
  │    07_price_trend_by_model.png                                  │
  │    08_co2_vs_deliveries.png                                     │
  │    09_model_r2_comparison.png                                   │
  │    10_model_rmse_comparison.png                                 │
  │    11_actual_vs_predicted.png                                   │
  │    12_feature_importance.png                                    │
  │    13_sarima_forecast.png                                       │
  │    14_prophet_forecast.png                                      │
  │    15_prophet_components.png                                    │
  │    16_ts_model_comparison.png                                   │
  └─────────────────────────────────────────────────────────────────┘
""")

print("  Pipeline complete ✓")

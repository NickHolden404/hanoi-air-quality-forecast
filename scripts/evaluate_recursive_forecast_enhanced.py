"""
Rolling-Origin Recursive PM2.5 Forecasting (FIXED)

"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    # Lag features
    for lag in [1,2,3,6,12,24]:
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

    # Rolling stats
    for w in [3,6,12,24]:
        df[f"pm25_roll_mean_{w}h"] = df["pm25"].rolling(w).mean()
        df[f"pm25_roll_std_{w}h"] = df["pm25"].rolling(w).std()
        df[f"pm25_roll_min_{w}h"] = df["pm25"].rolling(w).min()
        df[f"pm25_roll_max_{w}h"] = df["pm25"].rolling(w).max()

    # Temporal
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    # Trends
    for w in [6,12,24]:
        df[f"pm25_change_{w}h"] = df["pm25"].diff(w)

    df = df.dropna()
    return df

# ============================================================
# RECURSIVE FORECAST
# ============================================================

def recursive_forecast(model, x0, feature_cols, steps, history_df):
    preds = []
    recent = list(history_df["pm25"].tail(30).values)
    x = x0.copy()

    for _ in range(steps):
        y_hat = model.predict(x.reshape(1, -1))[0]
        preds.append(y_hat)
        recent.append(y_hat)
        recent = recent[-30:]

        for lag in [1,2,3,6,12,24]:
            col = f"pm25_lag_{lag}h"
            if col in feature_cols:
                x[feature_cols.index(col)] = recent[-lag]

        for w in [3,6,12,24]:
            if w <= len(recent):
                window = recent[-w:]
                stats = {
                    f"pm25_roll_mean_{w}h": np.mean(window),
                    f"pm25_roll_std_{w}h": np.std(window),
                    f"pm25_roll_min_{w}h": np.min(window),
                    f"pm25_roll_max_{w}h": np.max(window),
                }
                for col, val in stats.items():
                    if col in feature_cols:
                        x[feature_cols.index(col)] = val

    return np.array(preds)

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*80)
    print("ROLLING-ORIGIN RECURSIVE PM2.5 FORECASTING (FIXED)")
    print("="*80)

    df = pd.read_csv(
        "data/processed/hanoi_aqi_ml_ready.csv",
        parse_dates=["datetime"],
        index_col="datetime"
    )

    df = engineer_features(df)

    # HANDLE CATEGORICAL VARIABLES (THE BUG FIX)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    feature_cols = [c for c in df.columns if c != "pm25"]

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100, max_depth=7, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )

    horizons = [1,3,6,12,24]
    tscv = TimeSeriesSplit(n_splits=5)

    results = {
        m: {h: {"se": [], "ae": [], "yt": [], "yp": []} for h in horizons}
        for m in models
    }

    for fold, (tr, te) in enumerate(tscv.split(df), 1):
        print(f"\nFold {fold}")
        train, test = df.iloc[tr], df.iloc[te]

        X_train = train[feature_cols].values
        y_train = train["pm25"].values
        X_test = test[feature_cols].values
        y_test = test["pm25"].values

        fitted = {m: model.fit(X_train, y_train) for m, model in models.items()}

        test_points = range(0, len(test) - max(horizons), 24)

        for name, model in fitted.items():
            for h in horizons:
                for i in test_points:
                    preds = recursive_forecast(
                        model,
                        X_test[i],
                        feature_cols,
                        h,
                        df.iloc[:tr[-1] + i]
                    )
                    yt = y_test[i + h]
                    yp = preds[-1]

                    results[name][h]["se"].append((yt - yp)**2)
                    results[name][h]["ae"].append(abs(yt - yp))
                    results[name][h]["yt"].append(yt)
                    results[name][h]["yp"].append(yp)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for name in results:
        print(f"\n{name.upper()}")
        print(f"{'H':<6}{'RMSE':<10}{'MAE':<10}{'R²':<10}")
        for h in horizons:
            rmse = np.sqrt(np.mean(results[name][h]["se"]))
            mae = np.mean(results[name][h]["ae"])
            r2 = r2_score(results[name][h]["yt"], results[name][h]["yp"])
            print(f"{h:<6}{rmse:<10.2f}{mae:<10.2f}{r2:<10.3f}")

if __name__ == "__main__":

    main()

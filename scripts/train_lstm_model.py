"""
Direct Multi-Horizon LSTM / GRU Forecasting
Rolling-Origin Evaluation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# SEQUENCE CREATION
# =============================================================================

def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + horizon - 1, 0])  # PM2.5
    return np.array(X), np.array(y)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(df):
    feature_cols = ["pm25"]

    weather = ["TEMP", "DEWP", "PRES", "HUMI", "PRCP", "RAIN", "CLOUD", "SPRES"]
    feature_cols += [c for c in weather if c in df.columns]

    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    feature_cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    return df[feature_cols].values, feature_cols

# =============================================================================
# MODELS
# =============================================================================

def build_lstm(seq_len, n_feat):
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(seq_len, n_feat)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(seq_len, n_feat):
    model = keras.Sequential([
        layers.GRU(64, return_sequences=True, input_shape=(seq_len, n_feat)),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# =============================================================================
# METRICS
# =============================================================================

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("DIRECT MULTI-HORIZON LSTM / GRU FORECASTING")
    print("=" * 80)

    df = pd.read_csv(
        "data/processed/hanoi_aqi_ml_ready.csv",
        parse_dates=["datetime"],
        index_col="datetime"
    )

    rename = {
        "temperature": "TEMP",
        "humidity": "HUMI",
        "dew_point": "DEWP",
        "pressure_msl": "PRES",
        "precipitation": "PRCP",
        "rain": "RAIN",
        "cloud_cover": "CLOUD",
        "surface_pressure": "SPRES"
    }
    df = df.rename(columns=rename)

    data, feature_cols = prepare_data(df)

    seq_len = 24
    horizons = [1, 3, 6, 12, 24]
    n_folds = 5

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    print(f"\nSamples: {len(data):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Sequence length: {seq_len}")

    results = {
        "LSTM": {h: [] for h in horizons},
        "GRU":  {h: [] for h in horizons},
    }

    min_train = int(len(data) * 0.6)
    step = int((len(data) - min_train) / n_folds)

    for fold in range(n_folds):
        print(f"\n🔁 Fold {fold + 1}/{n_folds}")

        train_end = min_train + fold * step

        for h in horizons:
            X, y = create_sequences(data, seq_len, h)
            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:], y[train_end:]

            for name, builder in [("LSTM", build_lstm), ("GRU", build_gru)]:
                model = builder(seq_len, X.shape[2])

                model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=64,
                    verbose=0
                )

                pred = model.predict(X_test, verbose=0).flatten()

                dummy = np.zeros((len(pred), data.shape[1]))
                dummy[:, 0] = pred
                y_pred = scaler.inverse_transform(dummy)[:, 0]

                dummy[:, 0] = y_test
                y_true = scaler.inverse_transform(dummy)[:, 0]

                results[name][h].append(metrics(y_true, y_pred))

    print("\n" + "=" * 80)
    print("FINAL AVERAGED RESULTS")
    print("=" * 80)

    for model in ["LSTM", "GRU"]:
        print(f"\n📊 {model}")
        for h in horizons:
            rmse, mae, r2 = np.mean(results[model][h], axis=0)
            print(f"{h:>2}h → RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")

    print("\n✅ DONE")

if __name__ == "__main__":
    main()
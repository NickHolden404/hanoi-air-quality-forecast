import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict

print("=" * 80)
print("HANOI PM2.5 PREDICTION - ROLLING ORIGIN TRAINING")
print("=" * 80)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = pd.read_csv('data/processed/hanoi_aqi_ml_ready_fixed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\n✓ Loaded {len(df):,} records")

# -------------------------------------------------------------------
# Define features
# -------------------------------------------------------------------
temporal = ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_dry_season']
lag = [
    'pm25_lag1', 'pm25_lag24', 'pm25_lag168',
    'pm25_rolling_3h', 'pm25_rolling_24h',
    'pm25_rolling_7d', 'pm25_rolling_24h_std'
]
weather = [
    'temperature', 'humidity', 'dew_point', 'temp_humidity',
    'precipitation', 'rain', 'is_raining',
    'pressure_msl', 'surface_pressure', 'pressure_diff',
    'cloud_cover', 'wind_speed', 'wind_u', 'wind_v', 'wind_gusts'
]

all_features = temporal + lag + weather
weather_only = temporal + weather

X_all = df[all_features]
X_weather = df[weather_only]
y = df['pm25']

# -------------------------------------------------------------------
# Rolling-origin split helper
# -------------------------------------------------------------------
def rolling_origin_splits(n_samples, initial_train_size, test_size, step):
    splits = []
    start = initial_train_size
    while start + test_size <= n_samples:
        splits.append((slice(0, start), slice(start, start + test_size)))
        start += step
    return splits

# -------------------------------------------------------------------
# Rolling validation configuration
# -------------------------------------------------------------------
N = len(df)
INITIAL_TRAIN = int(0.7 * N)
TEST_WINDOW = 24 * 30   # 30 days
STEP = TEST_WINDOW

splits = rolling_origin_splits(N, INITIAL_TRAIN, TEST_WINDOW, STEP)
print(f"\n✓ Using {len(splits)} rolling folds")

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
models = {
    'Linear (Weather)': LinearRegression(),
    'Linear (Full)': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=5, random_state=42
    )
}

# -------------------------------------------------------------------
# Rolling evaluation
# -------------------------------------------------------------------
results = defaultdict(list)

for fold, (train_idx, test_idx) in enumerate(splits, 1):
    print(f"\n========== Fold {fold} ==========")

    X_train_all = X_all.iloc[train_idx]
    X_test_all = X_all.iloc[test_idx]
    X_train_weather = X_weather.iloc[train_idx]
    X_test_weather = X_weather.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # --- Persistence baseline ---
    y_persist = y_test.shift(1).dropna()
    y_true = y_test.loc[y_persist.index]

    rmse = np.sqrt(mean_squared_error(y_true, y_persist))
    mae = mean_absolute_error(y_true, y_persist)
    r2 = r2_score(y_true, y_persist)

    results['Model'].append('Persistence')
    results['Fold'].append(fold)
    results['RMSE'].append(rmse)
    results['MAE'].append(mae)
    results['R2'].append(r2)

    # --- ML models ---
    for name, model in models.items():
        print(f"Training {name}...")

        if name == 'Linear (Weather)':
            model.fit(X_train_weather, y_train)
            preds = model.predict(X_test_weather)
        else:
            model.fit(X_train_all, y_train)
            preds = model.predict(X_test_all)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results['Model'].append(name)
        results['Fold'].append(fold)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['R2'].append(r2)

# -------------------------------------------------------------------
# Aggregate results
# -------------------------------------------------------------------
results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby('Model')
    .agg(
        RMSE_mean=('RMSE', 'mean'),
        RMSE_std=('RMSE', 'std'),
        MAE_mean=('MAE', 'mean'),
        MAE_std=('MAE', 'std'),
        R2_mean=('R2', 'mean'),
        R2_std=('R2', 'std'),
    )
    .sort_values('RMSE_mean')
)

print("\n" + "=" * 80)
print("ROLLING ORIGIN VALIDATION SUMMARY")
print("=" * 80)
print(summary.round(3))
print("=" * 80)
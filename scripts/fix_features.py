import pandas as pd
import numpy as np

print("=" * 80)
print("FIXING DATA LEAKAGE - REBUILDING FEATURES")
print("=" * 80)

# Load the merged dataset
df = pd.read_csv('data/processed/hanoi_aqi_ml_dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\nLoaded {len(df):,} records")

# Sort by datetime
df = df.sort_values('datetime').reset_index(drop=True)

print("\n🔧 Rebuilding lag and rolling features WITHOUT leakage...")

# LAG FEATURES - Use .shift() to get PAST values only
df['pm25_lag1'] = df['pm25'].shift(1)
df['pm25_lag24'] = df['pm25'].shift(24)
df['pm25_lag168'] = df['pm25'].shift(168)

# ROLLING FEATURES - Calculate on SHIFTED data (past values only)
df['pm25_rolling_3h'] = df['pm25'].shift(1).rolling(window=3, min_periods=1).mean()
df['pm25_rolling_24h'] = df['pm25'].shift(1).rolling(window=24, min_periods=12).mean()
df['pm25_rolling_7d'] = df['pm25'].shift(1).rolling(window=168, min_periods=84).mean()
df['pm25_rolling_24h_std'] = df['pm25'].shift(1).rolling(window=24, min_periods=12).std()

print("   ✓ Lag features rebuilt")
print("   ✓ Rolling features rebuilt")

# WEATHER FEATURES
df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))
df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
df['pressure_diff'] = df['pressure_msl'].diff()
df['is_raining'] = (df['rain'] > 0).astype(int)

# VERIFICATION
print("\n🔍 Verifying no leakage...")
row3_pm25 = df.loc[3, 'pm25']
row3_rolling = df.loc[3, 'pm25_rolling_3h']
manual_calc = df.loc[0:2, 'pm25'].mean()

print(f"   Current PM2.5 (row 3): {row3_pm25}")
print(f"   pm25_rolling_3h: {row3_rolling:.2f}")
print(f"   Manual (avg rows 0-2): {manual_calc:.2f}")
print(f"   Match: {abs(row3_rolling - manual_calc) < 0.01}")

# Clean up NaN
print(f"\n🧹 Cleaning...")
print(f"   Before: {len(df):,}")
df = df.dropna()
print(f"   After: {len(df):,}")

# Save
df.to_csv('data/processed/hanoi_aqi_ml_ready_fixed.csv', index=False)

print("\n" + "=" * 80)
print("✅ FEATURES FIXED - NO DATA LEAKAGE")
print("=" * 80)
print(f"\n✓ Saved to: data/processed/hanoi_aqi_ml_ready_fixed.csv")
print(f"✓ Records: {len(df):,}")
print(f"✓ Features: {len(df.columns)}")
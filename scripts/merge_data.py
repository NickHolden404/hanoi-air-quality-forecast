import pandas as pd
import numpy as np

print("="*80)
print("MERGING HANOI AIR QUALITY & WEATHER DATA")
print("="*80)

# 1. LOAD WEATHER
print("\n1. Loading weather data...")
weather = pd.read_csv('data/raw/hanoi_weather_2014_2026.csv')
weather['datetime'] = pd.to_datetime(weather['datetime'])
print(f"   ✓ Loaded {len(weather):,} weather records")

# 2. LOAD PM2.5 - OPENAQ
print("\n2. Loading OpenAQ PM2.5...")
openaq = pd.read_csv('data/raw/hanoi_pm25_2years.csv')
openaq['datetime'] = pd.to_datetime(openaq['DateTime_Local'], utc=True).dt.tz_localize(None)
openaq = openaq[['datetime', 'PM2.5_Value']].copy()
openaq.rename(columns={'PM2.5_Value': 'pm25'}, inplace=True)
openaq['pm25'] = pd.to_numeric(openaq['pm25'], errors='coerce')
openaq['source'] = 'OpenAQ'
print(f"   ✓ {len(openaq):,} records")

# 3. LOAD PM2.5 - WAQI
print("\n3. Loading 5-year PM2.5...")
pm25_5y = pd.read_csv('data/raw/hanoi_pm25_5years.csv')
pm25_5y['datetime'] = pd.to_datetime(pm25_5y['DateTime_Local'], utc=True).dt.tz_localize(None)
pm25_5y = pm25_5y[['datetime', 'PM2.5_Value']].copy()
pm25_5y.rename(columns={'PM2.5_Value': 'pm25'}, inplace=True)
pm25_5y['pm25'] = pd.to_numeric(pm25_5y['pm25'], errors='coerce')
pm25_5y['source'] = '5-year'
print(f"   ✓ {len(pm25_5y):,} records")

# 4. COMBINE
print("\n4. Combining PM2.5...")
pm25_combined = pd.concat([openaq, pm25_5y], ignore_index=True)
print(f"   Total: {len(pm25_combined):,}")

# 5. CLEAN
print("\n5. Cleaning...")
pm25_combined = pm25_combined.dropna(subset=['datetime', 'pm25'])
pm25_combined = pm25_combined[(pm25_combined['pm25'] >= 0) & (pm25_combined['pm25'] <= 1000)]
print(f"   ✓ Clean: {len(pm25_combined):,}")

# 6. ROUND TO HOURLY
print("\n6. Rounding to hourly...")
pm25_combined['datetime'] = pm25_combined['datetime'].dt.round('h')
pm25_averaged = pm25_combined.groupby('datetime').agg({
    'pm25': 'mean',
    'source': lambda x: '+'.join(sorted(set(x))) if len(set(x)) > 1 else x.iloc[0]
}).reset_index()
print(f"   ✓ Averaged: {len(pm25_averaged):,}")

# 7. MERGE WITH WEATHER
print("\n7. Merging with weather...")
weather['datetime'] = weather['datetime'].dt.round('h')
merged = pd.merge(pm25_averaged, weather, on='datetime', how='inner')
print(f"   ✓ Merged: {len(merged):,}")

# 8. FEATURES
print("\n8. Feature engineering...")
merged = merged.sort_values('datetime').reset_index(drop=True)

# Temporal
merged['year'] = merged['datetime'].dt.year
merged['month'] = merged['datetime'].dt.month
merged['day'] = merged['datetime'].dt.day
merged['hour'] = merged['datetime'].dt.hour
merged['day_of_week'] = merged['datetime'].dt.dayofweek
merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
merged['season'] = merged['month'].apply(lambda x: 'dry' if x in [11,12,1,2,3,4] else 'wet')
merged['is_dry_season'] = (merged['season'] == 'dry').astype(int)

# Lag
merged['pm25_lag1'] = merged['pm25'].shift(1)
merged['pm25_lag24'] = merged['pm25'].shift(24)
merged['pm25_lag168'] = merged['pm25'].shift(168)
merged['pm25_rolling_3h'] = merged['pm25'].shift(1).rolling(3, min_periods=1).mean()
merged['pm25_rolling_24h'] = merged['pm25'].shift(1).rolling(24, min_periods=12).mean()
merged['pm25_rolling_7d'] = merged['pm25'].shift(1).rolling(168, min_periods=84).mean()
merged['pm25_rolling_24h_std'] = merged['pm25'].shift(1).rolling(24, min_periods=12).std()

# Weather
merged['wind_u'] = merged['wind_speed'] * np.cos(np.radians(merged['wind_direction']))
merged['wind_v'] = merged['wind_speed'] * np.sin(np.radians(merged['wind_direction']))
merged['temp_humidity'] = merged['temperature'] * merged['humidity'] / 100
merged['pressure_diff'] = merged['pressure_msl'].diff()
merged['is_raining'] = (merged['rain'] > 0).astype(int)

# Drop NaN
merged = merged.dropna()
print(f"   ✓ Final: {len(merged):,} records")

# 9. SAVE
print("\n9. Saving...")
merged.to_csv('data/processed/hanoi_aqi_ml_dataset.csv', index=False)
merged.drop('source', axis=1).to_csv('data/processed/hanoi_aqi_ml_ready.csv', index=False)

print("\n" + "="*80)
print("✅ SUCCESS!")
print(f"Final: {len(merged):,} records from {merged['datetime'].min()} to {merged['datetime'].max()}")
print("Files: data/processed/hanoi_aqi_ml_dataset.csv")
print("="*80)
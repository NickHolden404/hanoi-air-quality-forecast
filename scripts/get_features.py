import requests
import pandas as pd
from datetime import datetime

def download_hanoi_weather_historical():
    """
    Download historical weather data for Hanoi from Open-Meteo API (2014-2026)
    """
    
    print("Downloading historical weather data for Hanoi (2014-2026)...")
        
    # Hanoi coordinates
    latitude = 21.0285
    longitude = 105.8542
    
    # Date range
    start_date = "2014-01-01"
    end_date = "2026-01-28"
    
    # Open-Meteo Historical Weather API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters - all the important weather features for air quality prediction
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",           # Temperature at 2m height (°C)
            "relative_humidity_2m",     # Relative humidity (%)
            "dew_point_2m",            # Dew point (°C)
            "precipitation",            # Precipitation (mm)
            "rain",                     # Rain (mm)
            "pressure_msl",             # Atmospheric pressure at sea level (hPa)
            "surface_pressure",         # Surface pressure (hPa)
            "cloud_cover",              # Cloud cover (%)
            "wind_speed_10m",           # Wind speed at 10m (m/s)
            "wind_direction_10m",       # Wind direction (degrees)
            "wind_gusts_10m",          # Wind gusts (m/s)
        ],
        "timezone": "Asia/Bangkok"      # Vietnam timezone (UTC+7)
    }
    
    try:
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'dew_point': data['hourly']['dew_point_2m'],
            'precipitation': data['hourly']['precipitation'],
            'rain': data['hourly']['rain'],
            'pressure_msl': data['hourly']['pressure_msl'],
            'surface_pressure': data['hourly']['surface_pressure'],
            'cloud_cover': data['hourly']['cloud_cover'],
            'wind_speed': data['hourly']['wind_speed_10m'],
            'wind_direction': data['hourly']['wind_direction_10m'],
            'wind_gusts': data['hourly']['wind_gusts_10m'],
        })
        
        # Save to CSV
        filename = 'hanoi_weather_2014_2026.csv'
        df.to_csv(filename, index=False)
        
        print(f"✓ SUCCESS!")
        print(f"Downloaded {len(df)} hourly weather records")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Saved to: {filename}")
        print(f"\nColumns included:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Show sample
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error downloading weather data: {e}")
        return None

# Run the download
weather_df = download_hanoi_weather_historical()
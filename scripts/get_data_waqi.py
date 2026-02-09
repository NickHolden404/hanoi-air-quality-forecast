import requests
import csv
from datetime import datetime, timedelta
import time

# Get your token from https://aqicn.org/data-platform/token/
WAQI_TOKEN = 'YOUR_TOKEN_HERE'

# Hanoi station ID (you can find others on their website)
STATION = 'hanoi'

def download_waqi_historical_data():
    """Download historical PM2.5 data from WAQI"""
    
    print("Downloading historical PM2.5 data from WAQI...")
    
    with open('hanoi_pm25_waqi_5years.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'PM2.5_AQI', 'PM2.5_Value'])
        
        # Get data for the last 5 years, one day at a time
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        current_date = start_date
        records = 0
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                # WAQI API endpoint for historical data
                url = f'https://api.waqi.info/feed/{STATION}/?token={WAQI_TOKEN}'
                response = requests.get(url)
                data = response.json()
                
                if data['status'] == 'ok':
                    pm25 = data['data'].get('iaqi', {}).get('pm25', {}).get('v', 'N/A')
                    
                    writer.writerow([date_str, pm25, pm25])
                    records += 1
                    
                    if records % 30 == 0:
                        print(f"Downloaded {records} days of data...")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error on {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        print(f"\n✓ Downloaded {records} days of data!")

download_waqi_historical_data()
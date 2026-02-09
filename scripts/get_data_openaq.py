from openaq import OpenAQ
import csv
from datetime import datetime, timedelta

client = OpenAQ(api_key='YOUR-API-KEY-HERE')

# First, get the station details to find the PM2.5 sensor ID
print("Getting station 2161292 details...")
response = client.locations.get(2161292)
location = response.results[0]  # Get the actual location from results

pm25_sensor_id = None
for sensor in location.sensors:
    if sensor.parameter.name == 'pm25':
        pm25_sensor_id = sensor.id
        break

if not pm25_sensor_id:
    print("Error: No PM2.5 sensor found!")
    exit()

print(f"Found PM2.5 sensor ID: {pm25_sensor_id}")
print(f"Station: {location.name}")
print(f"Data available from: {location.datetime_first.local}")
print(f"Data available until: {location.datetime_last.local}")
print("\nStarting download...\n")

# Open CSV file
with open('hanoi_pm25_2years.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['DateTime_Local', 'DateTime_UTC', 'PM2.5_Value', 'Units', 'Station'])
    
    # Download in 3-month chunks to avoid timeout
    start_date = datetime(2024, 1, 29)
    end_date = datetime.now()
    
    current_start = start_date
    total_records = 0
    
    while current_start < end_date:
        # Download 3 months at a time
        current_end = min(current_start + timedelta(days=90), end_date)
        
        print(f"\nDownloading data from {current_start.date()} to {current_end.date()}...")
        
        page = 1
        while True:
            try:
                measurements = client.measurements.list(
                    sensors_id=pm25_sensor_id,
                    datetime_from=current_start.isoformat(),
                    datetime_to=current_end.isoformat(),
                    limit=1000,
                    page=page
                )
                
                # Write data
                for m in measurements.results:
                    writer.writerow([
                        m.period.datetime_to.local,
                        m.period.datetime_to.utc,
                        m.value,
                        m.parameter.units,
                        location.name
                    ])
                
                records_this_page = len(measurements.results)
                total_records += records_this_page
                print(f"  Page {page}: +{records_this_page} records (Total: {total_records})")
                
                if records_this_page < 1000:
                    break
                    
                page += 1
                
            except Exception as e:
                print(f"  Error on page {page}: {e}")
                break
        
        current_start = current_end

client.close()

print(f"\n✓ COMPLETE! Downloaded {total_records} PM2.5 measurements")
print(f"Data saved to hanoi_pm25_2years.csv")
print(f"Date range: January 2024 to {datetime.now().strftime('%B %Y')}")
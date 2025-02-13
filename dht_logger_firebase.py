import time
import board
import adafruit_dht
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime, timedelta

# Initialize Firebase with your specific credentials file
cred = credentials.Certificate('doll-house-bc1ea-firebase-adminsdk-rzwfu-bf53f1fcc7.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize DHT22 sensors with appropriate GPIO pins
sensors = {
    "Sensor 1": adafruit_dht.DHT22(board.D4),
    "Sensor 2": adafruit_dht.DHT22(board.D19),
    "Sensor 3": adafruit_dht.DHT22(board.D21)
}

def log_to_firebase(sensor_id, temperature, humidity):
    now = datetime.now()
    data = {
        'sensor_id': sensor_id,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S'),
        'temperature': round(temperature, 2),
        'humidity': round(humidity, 2)
    }
    # Add data to Firebase
    db.collection('sensor_readings').add(data)
    print(f"Data logged to Firebase for {sensor_id}: {data}")

# Function to read sensor data and handle errors
def read_sensor(sensor_id, sensor):
    try:
        temperature = sensor.temperature
        humidity = sensor.humidity
        if temperature is not None and humidity is not None:
            log_to_firebase(sensor_id, temperature, humidity)
        else:
            print(f"{sensor_id}: Failed to get valid data (None received).")
    except RuntimeError as error:
        print(f"{sensor_id}: Error - {error.args[0]}")
    except Exception as e:
        print(f"{sensor_id}: Unexpected error - {e}")

# Main data collection loop
start_time = datetime.now()
end_time = start_time + timedelta(hours=1)  # Run for 1 hour
interval = 20  # Log every 2 minutes (in seconds)

try:
    while datetime.now() < end_time:
        print(f"\nLogging data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")
        for sensor_id, sensor in sensors.items():
            read_sensor(sensor_id, sensor)
            time.sleep(2)  # Ensure at least 2 seconds between readings for each sensor
        
        # Wait for the remaining time in the interval
        time.sleep(interval - (len(sensors) * 2))

    print("\nData collection completed! Duration: 1 hour.")

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    # Safely release resources
    for sensor_id, sensor in sensors.items():
        sensor.exit()
    print("All sensors have been safely released.")

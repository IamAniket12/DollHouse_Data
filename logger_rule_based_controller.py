import time
import board
import RPi.GPIO as GPIO
import adafruit_dht
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime


class FloorController:
    def __init__(self, name, setpoint=22):
        self.name = name
        self.setpoint = setpoint
        self.tolerance = 0.5  # ±0.5°C tolerance
        self.last_action_time = 0
        self.min_action_interval = 30  # 30 seconds between actions

    def get_actions(self, current_temp):
        if current_temp is None:
            return None, None

        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval:
            return None, None

        self.last_action_time = current_time

        if current_temp < self.setpoint - self.tolerance:
            return True, False  # Turn on bulb, close window
        elif current_temp > self.setpoint + self.tolerance:
            return False, True  # Turn off bulb, open window
        else:
            return None, None  # Within acceptable range


class DollhouseSystem:
    def __init__(self):
        # Initialize Firebase
        cred = credentials.Certificate(
            "doll-house-bc1ea-firebase-adminsdk-rzwfu-bf53f1fcc7.json"
        )
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        # Initialize floor controllers
        self.controllers = {
            "top_floor": FloorController("Top Floor", setpoint=22),
            "ground_floor": FloorController("Ground Floor", setpoint=22),
        }

        # Initialize sensors
        self.sensors = {
            "external": adafruit_dht.DHT22(board.D22),
            "top_floor": adafruit_dht.DHT22(board.D27),
            "ground_floor": adafruit_dht.DHT22(board.D17),
        }

        # Initialize device states
        self.window_states = {
            "top_floor": False,
            "ground_floor": False,
        }  # False = closed
        self.bulb_states = {"top_floor": False, "ground_floor": False}  # False = off

        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        self.setup_gpio()

        self.logging_interval = 30  # 30 seconds

    def setup_gpio(self):
        # Setup bulb relay pins
        self.relay_pins = {"top_floor": 21, "ground_floor": 20}
        for pin in self.relay_pins.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)  # Initialize bulbs as OFF

        # Setup servo pins
        self.servo_pins = {"top_floor": 19, "ground_floor": 13}
        self.servos = {}
        for location, pin in self.servo_pins.items():
            GPIO.setup(pin, GPIO.OUT)
            self.servos[location] = GPIO.PWM(pin, 50)
            self.servos[location].start(0)

    def control_window(self, location, should_open):
        if should_open == self.window_states[location]:
            return

        servo = self.servos[location]
        angle = 90 if should_open else 0
        duty = angle / 18 + 2

        servo.ChangeDutyCycle(duty)
        time.sleep(0.5)
        servo.ChangeDutyCycle(0)  # Stop pulse

        self.window_states[location] = should_open
        print(f"{location} window {'opened' if should_open else 'closed'}")

    def control_bulb(self, location, should_turn_on):
        if should_turn_on == self.bulb_states[location]:
            return

        GPIO.output(
            self.relay_pins[location], not should_turn_on
        )  # Relay is active LOW
        self.bulb_states[location] = should_turn_on
        print(f"{location} bulb {'ON' if should_turn_on else 'OFF'}")

    def read_sensors(self):
        readings = {}
        for location, sensor in self.sensors.items():
            try:
                temperature = sensor.temperature
                humidity = sensor.humidity
                if temperature is not None and humidity is not None:
                    readings[location] = {
                        "temperature": round(temperature, 2),
                        "humidity": round(humidity, 2),
                    }
            except Exception as e:
                print(f"Error reading {location} sensor: {e}")
                readings[location] = None
            time.sleep(2)
        return readings

    def log_to_firebase(self, readings, timestamp):
        try:
            # Prepare data in flat structure
            data = {
                "timestamp": timestamp,
                "externalTemperature": readings.get("external", {}).get("temperature"),
                "externalHumidity": readings.get("external", {}).get("humidity"),
                "topFloorTemperature": readings.get("top_floor", {}).get("temperature"),
                "topFloorHumidity": readings.get("top_floor", {}).get("humidity"),
                "groundFloorTemperature": readings.get("ground_floor", {}).get(
                    "temperature"
                ),
                "groundFloorHumidity": readings.get("ground_floor", {}).get("humidity"),
                "topFloorBulbStatus": self.bulb_states.get("top_floor", False),
                "groundFloorBulbStatus": self.bulb_states.get("ground_floor", False),
                "topFloorWindowStatus": self.window_states.get("top_floor", False),
                "groundFloorWindowStatus": self.window_states.get(
                    "ground_floor", False
                ),
                "topFloorSetpoint": self.controllers["top_floor"].setpoint,
                "groundFloorSetpoint": self.controllers["ground_floor"].setpoint,
            }

            # Add data to Firestore
            self.db.collection("sensor_logs").add(data)
            print(f"\nData logged at {timestamp}")
            self.print_status(readings)

        except Exception as e:
            print(f"Error logging to Firebase: {e}")

    def print_status(self, readings):
        print("\nCurrent Status:")
        print("==============")
        for location in ["external", "top_floor", "ground_floor"]:
            if location in readings and readings[location]:
                print(f"{location.replace('_', ' ').title()}:")
                print(f"  Temperature: {readings[location]['temperature']}\u00b0C")
                print(f"  Humidity: {readings[location]['humidity']}%")
                if location != "external":
                    print(f"  Setpoint: {self.controllers[location].setpoint}\u00b0C")
                    print(
                        f"  Window: {'OPEN' if self.window_states[location] else 'CLOSED'}"
                    )
                    print(f"  Bulb: {'ON' if self.bulb_states[location] else 'OFF'}")
        print("==============")

    def run(self):
        print("Starting Automated Dollhouse Control System...")
        print("Temperature setpoint: 22°C for both floors")
        print(f"Logging interval: {self.logging_interval / 60} minutes")

        readings = self.read_sensors()
        print("Initial readings:", readings)
        self.log_to_firebase(readings, datetime.now())
        last_log_time = time.time()

        try:
            while True:
                current_time = time.time()
                readings = self.read_sensors()
                print("readings", readings)

                for location in ["top_floor", "ground_floor"]:
                    if location in readings and readings[location]:
                        temp = readings[location]["temperature"]
                        turn_on_bulb, open_window = self.controllers[
                            location
                        ].get_actions(temp)

                        if turn_on_bulb is not None:
                            self.control_bulb(location, turn_on_bulb)
                        if open_window is not None:
                            self.control_window(location, open_window)

                if current_time - last_log_time >= self.logging_interval:
                    self.log_to_firebase(readings, datetime.now())
                    last_log_time = current_time

                time.sleep(10)

        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        for sensor in self.sensors.values():
            sensor.exit()
        for servo in self.servos.values():
            servo.stop()
        GPIO.cleanup()
        print("System shutdown complete")


if __name__ == "__main__":
    system = DollhouseSystem()
    system.run()

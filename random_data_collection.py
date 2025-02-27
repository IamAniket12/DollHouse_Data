import time
import board
import random
import RPi.GPIO as GPIO
import adafruit_dht
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime


class RandomDataCollector:
    def __init__(self):
        # Initialize Firebase
        cred = credentials.Certificate(
            "doll-house-bc1ea-firebase-adminsdk-rzwfu-bf53f1fcc7.json"
        )
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        # Temperature safety limits
        self.min_safe_temp = 15.0  # Minimum safe temperature (°C)
        self.max_safe_temp = 35.0  # Maximum safe temperature (°C)

        # Random action parameters
        self.random_action_interval = 30  # Change random actions every 30 seconds
        self.last_random_action_time = 0

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

        # Setpoints (will be used for safety override only)
        self.setpoints = {"top_floor": 25, "ground_floor": 25}

        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        self.setup_gpio()

        self.logging_interval = 30  # 30 seconds
        self.retry_attempts = 3  # Number of sensor reading retry attempts

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
        current_angle = 90 if self.window_states[location] else 0
        target_angle = 90 if should_open else 0

        # Determine direction of movement
        step = 1 if target_angle > current_angle else -1

        # Gradually move the servo
        for angle in range(current_angle, target_angle + step, step):
            duty = angle / 18 + 2
            servo.ChangeDutyCycle(duty)
            time.sleep(0.05)  # Small delay between increments

        # Stop the pulse
        servo.ChangeDutyCycle(0)

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
            for attempt in range(self.retry_attempts):
                try:
                    temperature = sensor.temperature
                    humidity = sensor.humidity
                    if temperature is not None and humidity is not None:
                        readings[location] = {
                            "temperature": round(temperature, 2),
                            "humidity": round(humidity, 2),
                        }
                        break  # Success, exit retry loop
                except Exception as e:
                    print(
                        f"Error reading {location} sensor (attempt {attempt+1}/{self.retry_attempts}): {e}"
                    )
                    if attempt == self.retry_attempts - 1:  # Last attempt failed
                        readings[location] = None
                    else:
                        time.sleep(2)  # Wait before retry
            time.sleep(2)  # Time between different sensors
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
                "topFloorSetpoint": self.setpoints["top_floor"],
                "groundFloorSetpoint": self.setpoints["ground_floor"],
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
                    print(f"  Setpoint: {self.setpoints[location]}\u00b0C")
                    print(
                        f"  Window: {'OPEN' if self.window_states[location] else 'CLOSED'}"
                    )
                    print(f"  Bulb: {'ON' if self.bulb_states[location] else 'OFF'}")
        print("==============")

    def is_temperature_safe(self, location, readings):
        """Check if temperature is within safe range"""
        if location not in readings or readings[location] is None:
            return True  # Assume safe if no reading (will rely on last known state)

        temp = readings[location]["temperature"]
        return self.min_safe_temp <= temp <= self.max_safe_temp

    def safety_override(self, location, temp):
        """Apply safety actions if temperature is outside safe range"""
        if temp < self.min_safe_temp:
            # Too cold: turn on heater, close window
            print(
                f"SAFETY: {location} temperature {temp}°C below minimum {self.min_safe_temp}°C"
            )
            return True, False
        elif temp > self.max_safe_temp:
            # Too hot: turn off heater, open window
            print(
                f"SAFETY: {location} temperature {temp}°C above maximum {self.max_safe_temp}°C"
            )
            return False, True

        # Temperature is safe, no override needed
        return None, None

    def apply_random_actions(self, readings):
        """Apply random control actions if temperatures are within safe ranges"""
        for location in ["top_floor", "ground_floor"]:
            if location in readings and readings[location]:
                temp = readings[location]["temperature"]

                # Check if temperature is outside safe range
                if temp < self.min_safe_temp or temp > self.max_safe_temp:
                    # Apply safety override
                    bulb_action, window_action = self.safety_override(location, temp)
                    if bulb_action is not None:
                        self.control_bulb(location, bulb_action)
                    if window_action is not None:
                        self.control_window(location, window_action)
                else:
                    # Temperature is safe, apply random actions
                    # Bias bulb action based on current temperature
                    if temp < 20:  # Cold range
                        bulb_prob = 0.7  # 70% chance to turn on bulb when cold
                    elif temp > 30:  # Hot range
                        bulb_prob = 0.2  # 20% chance to turn on bulb when hot
                    else:  # Normal range
                        bulb_prob = 0.5  # 50% chance either way

                    random_bulb = random.random() < bulb_prob
                    random_window = random.random() < 0.5

                    self.control_bulb(location, random_bulb)
                    self.control_window(location, random_window)

    def get_experiment_info(self):
        # Determine which experiment we're running based on time (changes every 20 minutes)
        current_time = time.time()
        minutes_elapsed = int((current_time - self.start_time) / 60) % 60

        if minutes_elapsed < 20:
            return "Random Actions"
        elif minutes_elapsed < 40:
            return "Step Change Test"
        else:
            return "Alternating Pattern"

    def run_experiment(self, duration_hours=24):
        print(f"Starting Random Data Collection for {duration_hours} hours...")
        print(
            f"Temperature safety range: {self.min_safe_temp}°C to {self.max_safe_temp}°C"
        )
        print(f"Random action interval: {self.random_action_interval} seconds")
        print(f"Logging interval: {self.logging_interval} seconds")

        self.start_time = time.time()
        end_time = self.start_time + (duration_hours * 3600)
        last_log_time = self.start_time
        last_action_time = self.start_time

        # Initial sensor readings and logging
        readings = self.read_sensors()
        self.log_to_firebase(readings, datetime.now())

        # Initial experiment pattern
        pattern_start_time = self.start_time
        current_pattern = 0  # 0: random, 1: step test, 2: alternating

        try:
            while time.time() < end_time:
                current_time = time.time()
                readings = self.read_sensors()

                # Determine which pattern to use (changes every 20 minutes)
                elapsed_minutes = int((current_time - pattern_start_time) / 60)
                if elapsed_minutes >= 60:  # Reset every hour
                    pattern_start_time = current_time
                    elapsed_minutes = 0

                current_pattern = (elapsed_minutes // 20) % 3

                # Apply appropriate pattern
                if current_pattern == 0:
                    # Random pattern - change actions every 30 seconds
                    if current_time - last_action_time >= self.random_action_interval:
                        print("\nApplying random control actions")
                        self.apply_random_actions(readings)
                        last_action_time = current_time

                elif current_pattern == 1:
                    # Step test pattern
                    step_minute = (elapsed_minutes % 20) // 5  # Changes every 5 minutes
                    if (current_time - last_action_time) >= 5:  # Check every 5 seconds
                        if step_minute == 0:
                            # All ON/CLOSED
                            for loc in ["top_floor", "ground_floor"]:
                                if self.is_temperature_safe(loc, readings):
                                    self.control_bulb(loc, True)
                                    self.control_window(loc, False)
                        elif step_minute == 1:
                            # All OFF/OPEN
                            for loc in ["top_floor", "ground_floor"]:
                                if self.is_temperature_safe(loc, readings):
                                    self.control_bulb(loc, False)
                                    self.control_window(loc, True)
                        elif step_minute == 2:
                            # Top ON, Ground OFF
                            if self.is_temperature_safe("top_floor", readings):
                                self.control_bulb("top_floor", True)
                                self.control_window("top_floor", False)
                            if self.is_temperature_safe("ground_floor", readings):
                                self.control_bulb("ground_floor", False)
                                self.control_window("ground_floor", True)
                        elif step_minute == 3:
                            # Top OFF, Ground ON
                            if self.is_temperature_safe("top_floor", readings):
                                self.control_bulb("top_floor", False)
                                self.control_window("top_floor", True)
                            if self.is_temperature_safe("ground_floor", readings):
                                self.control_bulb("ground_floor", True)
                                self.control_window("ground_floor", False)
                        last_action_time = current_time

                elif current_pattern == 2:
                    # Alternating pattern - change every minute
                    minute = int((current_time - pattern_start_time) / 60) % 2
                    if (current_time - last_action_time) >= 5:  # Check every 5 seconds
                        for loc in ["top_floor", "ground_floor"]:
                            if self.is_temperature_safe(loc, readings):
                                self.control_bulb(loc, minute == 0)
                                self.control_window(loc, minute == 1)
                        last_action_time = current_time

                # Safety check on all temperatures regardless of pattern
                for location in ["top_floor", "ground_floor"]:
                    if location in readings and readings[location]:
                        temp = readings[location]["temperature"]
                        if temp < self.min_safe_temp:
                            self.control_bulb(location, True)  # Turn on heat
                            self.control_window(location, False)  # Close window
                        elif temp > self.max_safe_temp:
                            self.control_bulb(location, False)  # Turn off heat
                            self.control_window(location, True)  # Open window

                # Log data at regular intervals
                if current_time - last_log_time >= self.logging_interval:
                    self.log_to_firebase(readings, datetime.now())
                    last_log_time = current_time
                    experiment_name = [
                        "Random Actions",
                        "Step Change Test",
                        "Alternating Pattern",
                    ][current_pattern]
                    print(f"Current experiment: {experiment_name}")

                time.sleep(10)

        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
        except Exception as e:
            print(f"\nError during experiment: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        # Safe shutdown: turn off all bulbs, close all windows
        for location in ["top_floor", "ground_floor"]:
            self.control_bulb(location, False)
            self.control_window(location, False)

        for sensor in self.sensors.values():
            sensor.exit()
        for servo in self.servos.values():
            servo.stop()
        GPIO.cleanup()
        print("System shutdown complete")


if __name__ == "__main__":
    collector = RandomDataCollector()
    # Run for 48 hours by default
    collector.run_experiment(duration_hours=48)

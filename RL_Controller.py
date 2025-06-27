import time
import board
import numpy as np
import RPi.GPIO as GPIO
import adafruit_dht
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import os
import json
from collections import deque

# Import Stable Baselines and our modules
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym


class DummyEnv(gym.Env):
    """
    A minimal dummy environment for VecNormalize loading.
    This is only used to load normalization parameters, not for actual simulation.
    """
    def __init__(self):
        # Define observation space to match your training environment
        self.observation_space = gym.spaces.Box(
            low=np.array([-10.0, -10.0, -30.0, 0, 0, 0, 0, 10.0, 10.0, 0.0, 0]),
            high=np.array([50.0, 50.0, 50.0, 1, 1, 1, 1, 35.0, 35.0, 23.0, 2880]),
            dtype=np.float32,
        )
        
        # Define action space
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])
        
    def reset(self, **kwargs):
        return np.zeros(11, dtype=np.float32), {}
    
    def step(self, action):
        return np.zeros(11, dtype=np.float32), 0.0, False, False, {}


class RealDollhousePPOController:
    def __init__(
        self, model_path, vec_normalize_path=None, env_params_path=None
    ):
        """
        Initialize the real dollhouse PPO controller.

        Args:
            model_path: Path to the trained PPO model
            vec_normalize_path: Path to VecNormalize parameters (optional)
            env_params_path: Path to environment parameters (optional)
        """
        # Initialize Firebase
        cred = credentials.Certificate(
            "doll-house-bc1ea-firebase-adminsdk-rzwfu-bf53f1fcc7.json"
        )
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        # Load environment parameters
        self.load_environment_params(env_params_path)

        # Load PPO model and normalization
        print(f"Loading PPO model from {model_path}...")
        self.ppo_model = PPO.load(model_path)

        # Load normalization if available
        self.vec_normalize = None
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            print(f"Loading normalization parameters from {vec_normalize_path}")
            try:
                # Create a proper dummy environment for VecNormalize
                dummy_env = DummyVecEnv([lambda: DummyEnv()])
                self.vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                print("Normalization loaded successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not load normalization: {e}")
                print("Proceeding without normalization")
                self.vec_normalize = None

        # Temperature safety limits
        self.min_safe_temp = 15.0  # Minimum safe temperature (°C)
        self.max_safe_temp = 35.0  # Maximum safe temperature (°C)

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

        # Control history for reference (not needed without SINDy)
        self.temp_history = {
            "ground_floor": deque(maxlen=3),
            "top_floor": deque(maxlen=3),
            "external": deque(maxlen=3),
        }

        # Action history
        self.action_history = deque(maxlen=3)

        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        self.setup_gpio()

        # Control parameters
        self.control_interval = 30  # PPO control every 30 seconds
        self.logging_interval = 30  # Log every 30 seconds
        self.retry_attempts = 3

        # Episode tracking
        self.episode_start_time = None
        self.step_count = 0
        self.total_reward = 0.0

    def load_environment_params(self, env_params_path):
        """Load environment parameters from file or use defaults."""
        if env_params_path and os.path.exists(env_params_path):
            with open(env_params_path, "r") as f:
                params = json.load(f)
            print(f"Loaded environment parameters from {env_params_path}")
        else:
            params = {}
            print("Using default environment parameters")

        # Set parameters with defaults
        self.heating_setpoint = params.get("heating_setpoint", 26.0)
        self.cooling_setpoint = params.get("cooling_setpoint", 28.0)
        self.time_step_seconds = params.get("time_step_seconds", 30)
        self.setpoint_pattern = params.get("setpoint_pattern", "fixed")
        self.reward_type = params.get("reward_type", "balanced")
        self.energy_weight = params.get("energy_weight", 0.5)
        self.comfort_weight = params.get("comfort_weight", 1.0)

        print(f"Setpoint range: {self.heating_setpoint}°C - {self.cooling_setpoint}°C")

    def setup_gpio(self):
        """Setup GPIO pins for relays and servos."""
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
        """Control window servo motor."""
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
            time.sleep(0.05)

        # Stop the pulse
        servo.ChangeDutyCycle(0)

        self.window_states[location] = should_open
        print(f"🪟 {location} window {'OPENED' if should_open else 'CLOSED'}")

    def control_bulb(self, location, should_turn_on):
        """Control bulb relay."""
        if should_turn_on == self.bulb_states[location]:
            return

        GPIO.output(
            self.relay_pins[location], not should_turn_on
        )  # Relay is active LOW
        self.bulb_states[location] = should_turn_on
        print(f"💡 {location} bulb {'ON' if should_turn_on else 'OFF'}")

    def read_sensors(self):
        """Read all temperature and humidity sensors."""
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
                        break
                except Exception as e:
                    print(
                        f"❌ Error reading {location} sensor (attempt {attempt+1}): {e}"
                    )
                    if attempt == self.retry_attempts - 1:
                        readings[location] = None
                    else:
                        time.sleep(2)
            time.sleep(1)  # Brief pause between sensors
        return readings

    def update_temperature_history(self, readings):
        """Update temperature history for reference."""
        for location in ["ground_floor", "top_floor", "external"]:
            if location in readings and readings[location]:
                temp = readings[location]["temperature"]
                self.temp_history[location].append(temp)
            else:
                # Use last known temperature if reading failed
                if self.temp_history[location]:
                    self.temp_history[location].append(self.temp_history[location][-1])
                else:
                    # Initialize with reasonable defaults
                    default_temps = {
                        "ground_floor": 22.0,
                        "top_floor": 23.0,
                        "external": 20.0,
                    }
                    self.temp_history[location].append(default_temps[location])

    def get_current_setpoints(self):
        """Get current heating and cooling setpoints based on pattern."""
        if self.setpoint_pattern == "fixed":
            return self.heating_setpoint, self.cooling_setpoint

        elif self.setpoint_pattern == "schedule":
            current_hour = datetime.now().hour
            if 9 <= current_hour < 18:  # Daytime
                return 22.0, 24.0
            elif 8 <= current_hour < 9:  # Morning
                return 26.0, 28.0
            else:  # Night
                return 20.0, 24.0

        else:
            return self.heating_setpoint, self.cooling_setpoint

    def create_observation(self, readings):
        """Create observation vector for PPO model."""
        # Get current temperatures
        ground_temp = readings.get("ground_floor", {}).get("temperature", 22.0)
        top_temp = readings.get("top_floor", {}).get("temperature", 23.0)
        external_temp = readings.get("external", {}).get("temperature", 20.0)

        # Get current setpoints
        heating_sp, cooling_sp = self.get_current_setpoints()

        # Calculate hour of day
        hour_of_day = datetime.now().hour + datetime.now().minute / 60.0

        # Create observation vector matching training format
        observation = np.array(
            [
                ground_temp,  # 0: ground_temp
                top_temp,  # 1: top_temp
                external_temp,  # 2: external_temp
                int(self.bulb_states["ground_floor"]),  # 3: prev_ground_light
                int(self.window_states["ground_floor"]),  # 4: prev_ground_window
                int(self.bulb_states["top_floor"]),  # 5: prev_top_light
                int(self.window_states["top_floor"]),  # 6: prev_top_window
                heating_sp,  # 7: heating_setpoint
                cooling_sp,  # 8: cooling_setpoint
                hour_of_day,  # 9: hour_of_day
                self.step_count,  # 10: time_step_in_episode
            ],
            dtype=np.float32,
        )

        return observation

    def get_ppo_action(self, observation):
        """Get action from PPO model."""
        try:
            # Apply normalization if available
            if self.vec_normalize:
                # Use the normalization directly on the observation
                normalized_obs = self.vec_normalize.normalize_obs(
                    observation.reshape(1, -1)
                )
                action, _ = self.ppo_model.predict(normalized_obs, deterministic=True)
            else:
                action, _ = self.ppo_model.predict(observation, deterministic=True)

            # Ensure action is in correct format
            if isinstance(action, np.ndarray):
                action = action.flatten()

            # Convert to integers for GPIO control
            action = [int(a) for a in action]

            return action

        except Exception as e:
            print(f"❌ Error getting PPO action: {e}")
            import traceback
            traceback.print_exc()
            # Return safe default action (all off/closed)
            return [0, 0, 0, 0]

    def apply_safety_override(self, action, readings):
        """Apply safety overrides if temperatures are dangerous."""
        safe_action = action.copy()

        for i, location in enumerate(["ground_floor", "top_floor"]):
            if location in readings and readings[location]:
                temp = readings[location]["temperature"]

                if temp < self.min_safe_temp:
                    # Too cold: force heater on, window closed
                    safe_action[i * 2] = 1  # Turn on bulb
                    safe_action[i * 2 + 1] = 0  # Close window
                    print(
                        f"🚨 SAFETY: {location} too cold ({temp}°C) - forcing heat on"
                    )

                elif temp > self.max_safe_temp:
                    # Too hot: force heater off, window open
                    safe_action[i * 2] = 0  # Turn off bulb
                    safe_action[i * 2 + 1] = 1  # Open window
                    print(f"🚨 SAFETY: {location} too hot ({temp}°C) - forcing cooling")

        return safe_action

    def calculate_reward(self, readings, action):
        """Calculate reward for current state and action."""
        try:
            ground_temp = readings.get("ground_floor", {}).get("temperature", 22.0)
            top_temp = readings.get("top_floor", {}).get("temperature", 23.0)
            heating_sp, cooling_sp = self.get_current_setpoints()

            # Comfort reward
            ground_comfort = 1.0 if heating_sp <= ground_temp <= cooling_sp else 0.0
            top_comfort = 1.0 if heating_sp <= top_temp <= cooling_sp else 0.0
            comfort_reward = (ground_comfort + top_comfort) / 2.0

            # Energy penalty
            lights_on = action[0] + action[2]  # Ground light + Top light
            energy_penalty = (lights_on / 2.0) * self.energy_weight

            # Total reward
            reward = self.comfort_weight * comfort_reward - energy_penalty

            return reward

        except Exception as e:
            print(f"❌ Error calculating reward: {e}")
            return 0.0

    def log_to_firebase(self, readings, action, reward, observation, timestamp):
        """Log all data to Firebase real_time_control collection."""
        try:
            # Helper function to convert numpy types to native Python types
            def convert_value(value):
                if value is None:
                    return None
                if isinstance(value, (np.integer, np.floating)):
                    return value.item()  # Converts numpy scalar to Python scalar
                if isinstance(value, np.ndarray):
                    return value.tolist()  # Convert arrays to lists
                if isinstance(value, bool):
                    return bool(value)  # Ensure it's Python bool, not numpy bool
                return value

            # Prepare comprehensive data with type conversion
            data = {
                # Timestamp and episode info
                "timestamp": timestamp,
                "episode_start_time": self.episode_start_time,
                "step_count": convert_value(self.step_count),
                "total_reward": convert_value(self.total_reward),
                
                # Sensor readings
                "externalTemperature": convert_value(readings.get("external", {}).get("temperature")),
                "externalHumidity": convert_value(readings.get("external", {}).get("humidity")),
                "topFloorTemperature": convert_value(readings.get("top_floor", {}).get("temperature")),
                "topFloorHumidity": convert_value(readings.get("top_floor", {}).get("humidity")),
                "groundFloorTemperature": convert_value(readings.get("ground_floor", {}).get("temperature")),
                "groundFloorHumidity": convert_value(readings.get("ground_floor", {}).get("humidity")),
                
                # PPO actions (what the model decided)
                "ppoGroundLightAction": convert_value(action[0]),
                "ppoGroundWindowAction": convert_value(action[1]),
                "ppoTopLightAction": convert_value(action[2]),
                "ppoTopWindowAction": convert_value(action[3]),
                
                # Actual device states (after safety override)
                "topFloorBulbStatus": convert_value(self.bulb_states.get("top_floor", False)),
                "groundFloorBulbStatus": convert_value(self.bulb_states.get("ground_floor", False)),
                "topFloorWindowStatus": convert_value(self.window_states.get("top_floor", False)),
                "groundFloorWindowStatus": convert_value(self.window_states.get("ground_floor", False)),
                
                # Setpoints and control parameters
                "heatingSetpoint": convert_value(observation[7]),
                "coolingSetpoint": convert_value(observation[8]),
                "hourOfDay": convert_value(observation[9]),
                
                # Performance metrics
                "stepReward": convert_value(reward),
                "groundComfortViolation": convert_value(max(0, observation[7] - observation[0]) + max(0, observation[0] - observation[8])),
                "topComfortViolation": convert_value(max(0, observation[7] - observation[1]) + max(0, observation[1] - observation[8])),
                "energyUse": convert_value(action[0] + action[2]),
                
                # Derived features for analysis
                "avgTemp": convert_value((observation[0] + observation[1]) / 2),
                "tempDifference": convert_value(observation[1] - observation[0]),
                "avgSetpoint": convert_value((observation[7] + observation[8]) / 2),
                "groundTempDeviation": convert_value(observation[0] - (observation[7] + observation[8]) / 2),
                "topTempDeviation": convert_value(observation[1] - (observation[7] + observation[8]) / 2),
                
                # Model info
                "controllerType": "ppo_model",
                "useNormalization": convert_value(self.vec_normalize is not None),
                "setpointPattern": self.setpoint_pattern,
                "rewardType": self.reward_type,
            }

            # Add to Firestore
            self.db.collection("real_time_control").add(data)
            print(f"📊 Data logged to Firebase at {timestamp}")

        except Exception as e:
            print(f"❌ Error logging to Firebase: {e}")
            import traceback
            traceback.print_exc()  # This will help debug any remaining issues

    def print_status(self, readings, action, reward, observation):
        """Print current system status."""
        print(f"\n{'='*60}")
        print(f"🤖 PPO DOLLHOUSE CONTROL - Step {self.step_count}")
        print(f"{'='*60}")

        # Temperature readings
        print("🌡️  TEMPERATURE READINGS:")
        for location in ["external", "ground_floor", "top_floor"]:
            if location in readings and readings[location]:
                temp = readings[location]["temperature"]
                humidity = readings[location]["humidity"]
                print(
                    f"   {location.replace('_', ' ').title():12}: {temp:5.1f}°C, {humidity:4.1f}%"
                )

        # Setpoints
        heating_sp, cooling_sp = self.get_current_setpoints()
        print(f"   Setpoint Range: {heating_sp:.1f}°C - {cooling_sp:.1f}°C")

        # PPO decision
        action_names = ["Ground Light", "Ground Window", "Top Light", "Top Window"]
        print("\n🧠 PPO MODEL DECISION:")
        for i, (name, act) in enumerate(zip(action_names, action)):
            status = "ON/OPEN" if act else "OFF/CLOSED"
            print(f"   {name:12}: {status}")

        # Actual device states
        print("\n⚡ ACTUAL DEVICE STATES:")
        print(
            f"   Ground Light : {'ON' if self.bulb_states['ground_floor'] else 'OFF'}"
        )
        print(
            f"   Ground Window: {'OPEN' if self.window_states['ground_floor'] else 'CLOSED'}"
        )
        print(f"   Top Light    : {'ON' if self.bulb_states['top_floor'] else 'OFF'}")
        print(
            f"   Top Window   : {'OPEN' if self.window_states['top_floor'] else 'CLOSED'}"
        )

        # Performance
        print(f"\n📈 PERFORMANCE:")
        print(f"   Step Reward  : {reward:6.3f}")
        print(f"   Total Reward : {self.total_reward:6.3f}")
        print(f"   Avg Reward   : {self.total_reward/max(1, self.step_count):6.3f}")
        print(f"   Normalization: {'Yes' if self.vec_normalize else 'No'}")

        print(f"{'='*60}")

    def run_control_experiment(self, duration_hours=24):
        """Run the PPO control experiment."""
        print(f"🚀 Starting PPO Dollhouse Control for {duration_hours} hours")
        print(
            f"📊 Model: PPO with {'normalization' if self.vec_normalize else 'no normalization'}"
        )
        print(f"🎯 Setpoints: {self.heating_setpoint}°C - {self.cooling_setpoint}°C")
        print(f"🔒 Safety range: {self.min_safe_temp}°C - {self.max_safe_temp}°C")
        print(f"⏱️  Control interval: {self.control_interval} seconds")

        self.episode_start_time = datetime.now()
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        last_control_time = start_time
        last_log_time = start_time

        # Initialize temperature history
        initial_readings = self.read_sensors()
        for _ in range(3):  # Fill history with initial readings
            self.update_temperature_history(initial_readings)
            self.action_history.append([0, 0, 0, 0])  # Initialize with all off

        try:
            while time.time() < end_time:
                current_time = time.time()

                # Read sensors
                readings = self.read_sensors()
                self.update_temperature_history(readings)

                # PPO control decision
                if current_time - last_control_time >= self.control_interval:
                    # Create observation for PPO
                    observation = self.create_observation(readings)

                    # Get PPO action
                    ppo_action = self.get_ppo_action(observation)

                    # Apply safety override if needed
                    safe_action = self.apply_safety_override(ppo_action, readings)

                    # Execute actions
                    self.control_bulb("ground_floor", bool(safe_action[0]))
                    self.control_window("ground_floor", bool(safe_action[1]))
                    self.control_bulb("top_floor", bool(safe_action[2]))
                    self.control_window("top_floor", bool(safe_action[3]))

                    # Calculate reward
                    reward = self.calculate_reward(readings, safe_action)
                    self.total_reward += reward
                    self.step_count += 1

                    # Store action in history
                    self.action_history.append(safe_action)

                    # Log to Firebase
                    self.log_to_firebase(
                        readings, safe_action, reward, observation, datetime.now()
                    )

                    # Print status
                    self.print_status(readings, safe_action, reward, observation)

                    last_control_time = current_time

                # Sleep briefly
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Experiment interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during experiment: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean shutdown of the system."""
        print("\n🧹 Cleaning up system...")

        # Safe shutdown: turn off all bulbs, close all windows
        for location in ["top_floor", "ground_floor"]:
            self.control_bulb(location, False)
            self.control_window(location, False)

        # Clean up GPIO
        for sensor in self.sensors.values():
            sensor.exit()
        for servo in self.servos.values():
            servo.stop()
        GPIO.cleanup()

        print("✅ System shutdown complete")
        print(f"📊 Final Statistics:")
        print(f"   Total Steps: {self.step_count}")
        print(f"   Total Reward: {self.total_reward:.3f}")
        print(f"   Average Reward: {self.total_reward/max(1, self.step_count):.3f}")


def main():
    """Main function to run PPO control."""
    import argparse

    parser = argparse.ArgumentParser(description="Real Dollhouse PPO Control")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO model")
    parser.add_argument("--vec-normalize-path", help="Path to VecNormalize parameters")
    parser.add_argument("--env-params", help="Path to environment parameters JSON")
    parser.add_argument("--duration", type=float, default=24, help="Duration in hours")

    args = parser.parse_args()

    try:
        # Create controller
        controller = RealDollhousePPOController(
            model_path=args.model_path,
            vec_normalize_path=args.vec_normalize_path,
            env_params_path=args.env_params,
        )

        # Run experiment
        controller.run_control_experiment(duration_hours=args.duration)

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


# Example usage:
# python RL_Controller.py \
#   --model-path "Controllers/PPO/ppo_final_model.zip" \
#   --vec-normalize-path "Controllers/PPO/vec_normalize.pkl" \
#   --env-params "Controllers/PPO/env_params.json" \
#   --duration 2
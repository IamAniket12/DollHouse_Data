import RPi.GPIO as GPIO
import time

# Pin configuration
RELAY_PIN2 = 21  # GPIO pin connected to IN1 (Bulb 1)
RELAY_PIN3 = 20  # GPIO pin connected to IN2 (Bulb 2)

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN2, GPIO.OUT)
GPIO.setup(RELAY_PIN3, GPIO.OUT)

# Turn off both relays initially
GPIO.output(RELAY_PIN2, GPIO.HIGH)
GPIO.output(RELAY_PIN3, GPIO.HIGH)

print("Control the bulbs using the following keys:")
print("1: Turn Bulb 1 ON")
print("2: Turn Bulb 1 OFF")
print("3: Turn Bulb 2 ON")
print("4: Turn Bulb 2 OFF")
print("q: Quit the program")

try:
    while True:
        key = input("Enter your choice: ")
        
        if key == "1":
            print("Turning Top Floor Bulb  ON")
            GPIO.output(RELAY_PIN2, GPIO.LOW)  # Turn ON Bulb 1
        elif key == "2":
            print("Turning Top Floor Bulb 1 OFF")
            GPIO.output(RELAY_PIN2, GPIO.HIGH)  # Turn OFF Bulb 1
        elif key == "3":
            print("Turning Ground Floor Bulb 2 ON")
            GPIO.output(RELAY_PIN3, GPIO.LOW)  # Turn ON Bulb 2
        elif key == "4":
            print("Turning Ground Floor Bulb 2  OFF")
            GPIO.output(RELAY_PIN3, GPIO.HIGH)  # Turn OFF Bulb 2
        elif key == "q":
            print("Exiting...")
            break
        else:
            print("Invalid input! Please press 1, 2, 3, 4, or q.")
        
except KeyboardInterrupt:
    print("\nProgram interrupted by the user.")

finally:
    print("Cleaning up GPIO...")
    GPIO.cleanup()


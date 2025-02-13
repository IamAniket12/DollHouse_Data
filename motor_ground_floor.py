import RPi.GPIO as GPIO
import time

# Set the GPIO mode to BCM (Broadcom SOC channel numbers)
GPIO.setmode(GPIO.BCM)

# Set GPIO1 as the pin for controlling the servo
servo_pin = 13
GPIO.setup(servo_pin, GPIO.OUT)

# Set PWM frequency to 50Hz (standard for servos)
pwm = GPIO.PWM(servo_pin, 50)

# Start PWM with 0% duty cycle (servo at 0 degree)
pwm.start(0)

# Function to set servo angle
def set_angle(angle):
    # Calculate duty cycle from angle
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)  # Delay for smoother movement

try:
    current_angle = 0  # Initialize angle at 0 degrees
    set_angle(current_angle)  # Move servo to 0 degrees at start
    print("Press 'O' to open (gradually to 90 degrees), 'C' to close (gradually to 0 degrees), or 'Q' to quit.")

    while True:
        user_input = input("Enter command: ").strip().lower()
        if user_input == 'o':  # Gradually increase angle to 90
            print("Opening window to 90 degrees...")
            while current_angle < 90:
                current_angle += 1
                set_angle(current_angle)
            print("Window fully opened at 90 degrees.")
        elif user_input == 'c':  # Gradually decrease angle to 0
            print("Closing window to 0 degrees...")
            while current_angle > 0:
                current_angle -= 1
                set_angle(current_angle)
            print("Window fully closed at 0 degrees.")
        elif user_input == 'q':  # Quit the program
            print("Quitting program.")
            break
        else:
            print("Invalid command. Use 'O', 'C', or 'Q'.")

finally:
    # Clean up and stop PWM
    pwm.stop()
    GPIO.cleanup()

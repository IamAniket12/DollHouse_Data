import RPi.GPIO as GPIO
import time

class ServoController:
    def __init__(self, pin, name):
        self.pin = pin
        self.name = name
        self.current_angle = 0
        
        # Setup GPIO for servo
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)  # 50Hz frequency
        self.pwm.start(0)
        
    def set_angle(self, angle):
        duty = angle / 18 + 2
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.02)
        
    def open_window(self):
        print(f"Opening {self.name} window to 90 degrees...")
        while self.current_angle < 90:
            self.current_angle += 1
            self.set_angle(self.current_angle)
        print(f"{self.name} window fully opened at 90 degrees.")
        
    def close_window(self):
        print(f"Closing {self.name} window to 0 degrees...")
        while self.current_angle > 0:
            self.current_angle -= 1
            self.set_angle(self.current_angle)
        print(f"{self.name} window fully closed at 0 degrees.")
        
    def cleanup(self):
        self.pwm.stop()

class RelayController:
    def __init__(self, pin, name):
        self.pin = pin
        self.name = name
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.HIGH)  # Initially OFF
        
    def turn_on(self):
        GPIO.output(self.pin, GPIO.LOW)
        print(f"{self.name} turned ON")
        
    def turn_off(self):
        GPIO.output(self.pin, GPIO.HIGH)
        print(f"{self.name} turned OFF")

class DollhouseController:
    def __init__(self):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        
        # Create servo controllers
        self.ground_floor_servo = ServoController(13, "Ground Floor")
        self.top_floor_servo = ServoController(19, "Top Floor")
        
        # Create relay controllers
        self.top_floor_bulb = RelayController(21, "Top Floor Bulb")
        self.ground_floor_bulb = RelayController(20, "Ground Floor Bulb")
        
        self.print_instructions()
    
    def print_instructions(self):
        print("\nDollhouse Control System")
        print("========================")
        print("Window Controls:")
        print("OG: Open Ground Floor Window")
        print("CG: Close Ground Floor Window")
        print("OT: Open Top Floor Window")
        print("CT: Close Top Floor Window")
        print("\nBulb Controls:")
        print("1: Turn Top Floor Bulb ON")
        print("2: Turn Top Floor Bulb OFF")
        print("3: Turn Ground Floor Bulb ON")
        print("4: Turn Ground Floor Bulb OFF")
        print("Q: Quit the program")
        print("========================")
    
    def run(self):
        try:
            while True:
                command = input("\nEnter command: ").strip().upper()
                
                if command == 'OG':
                    self.ground_floor_servo.open_window()
                elif command == 'CG':
                    self.ground_floor_servo.close_window()
                elif command == 'OT':
                    self.top_floor_servo.open_window()
                elif command == 'CT':
                    self.top_floor_servo.close_window()
                elif command == '1':
                    self.top_floor_bulb.turn_on()
                elif command == '2':
                    self.top_floor_bulb.turn_off()
                elif command == '3':
                    self.ground_floor_bulb.turn_on()
                elif command == '4':
                    self.ground_floor_bulb.turn_off()
                elif command == 'Q':
                    print("Exiting program...")
                    break
                else:
                    print("Invalid command!")
                    self.print_instructions()
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("Cleaning up...")
        self.ground_floor_servo.cleanup()
        self.top_floor_servo.cleanup()
        GPIO.cleanup()

if __name__ == "__main__":
    dollhouse = DollhouseController()
    dollhouse.run()

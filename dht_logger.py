# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_dht

# Initialize the DHT devices with data pins connected to:
dhtDevice = adafruit_dht.DHT22(board.D22)
dhtDevice1 = adafruit_dht.DHT22(board.D27)
dhtDevice2 = adafruit_dht.DHT22(board.D17)  # Added the third sensor

while True:
    try:
        # Read from the first sensor
        temperature_c = dhtDevice.temperature
        temperature_f = temperature_c * (9 / 5) + 32
        humidity = dhtDevice.humidity
        print(
            "Sensor 1 - Temp: {:.1f} F / {:.1f} C    Humidity: {}%".format(
                temperature_f, temperature_c, humidity
            )
        )

        # Read from the second sensor
        temperature_c1 = dhtDevice1.temperature
        temperature_f1 = temperature_c1 * (9 / 5) + 32
        humidity1 = dhtDevice1.humidity
        print(
            "Sensor 2 - Temp: {:.1f} F / {:.1f} C    Humidity: {}%".format(
                temperature_f1, temperature_c1, humidity1
            )
        )

        # Read from the third sensor
        temperature_c2 = dhtDevice2.temperature
        temperature_f2 = temperature_c2 * (9 / 5) + 32
        humidity2 = dhtDevice2.humidity
        print(
            "Sensor 3 - Temp: {:.1f} F / {:.1f} C    Humidity: {}%".format(
                temperature_f2, temperature_c2, humidity2
            )
        )

    except RuntimeError as error:
        # Handle sensor read errors
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        dhtDevice.exit()
        dhtDevice1.exit()
        dhtDevice2.exit()
        raise error

    time.sleep(2.0)

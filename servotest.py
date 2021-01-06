from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685



# Servo Channels
IRIS_SERVO = 4
DOME_LIGHTS = 1
IRIS_LIGHT = 2
HOVER_LIGHTS = 3

FREQUENCY = 50
PERIOD = 1.0 / float(FREQUENCY) * 1000.0

# create iris servo
i2c_bus = busio.I2C(SCL, SDA)
pca = PCA9685(i2c_bus)
pca.frequency = FREQUENCY

# Convenience Servo Values
ON = 1.0
OPEN = 1.0
MID = 0.5
CLOSED = 0.0
OFF = 0.0

def dalek_servo(channel,value):
    print("Channel: " + str(channel))
    value = value + 1.0
    print("Value: " + str(value))
    duty_cycle = int(value / PERIOD / 65535.0)
    print("Duty Cycle: " + str(duty_cycle))
    pca.channels[channel].duty_cycle = duty_cycle

while True:
    value = float(input("Value 0.0 to 1.0 - "))
    dalek_servo(IRIS_SERVO,value)
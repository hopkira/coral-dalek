from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# create iris servo
i2c_bus = busio.I2C(SCL, SDA)
pca = PCA9685(i2c_bus)
pca.frequency = 50

# Servo Channels
IRIS_SERVO = 4
DOME_LIGHTS = 1
IRIS_LIGHT = 2
HOVER_LIGHTS = 3

# Convenience Servo Values
ON = 0xffff
OPEN = 0x7FFF
MID = 0x7FFF
CLOSED = 0x7FFF
OFF = 0x0000

def dalek_servo(channel,value):
    value = ON * value
    pca.channels[channel].duty_cycle = value

def servo_state(instr_list):
    for instr in instr_list:
        dalek_servo(instr[0],instr[1])

servo_state([IRIS_SERVO, MID])
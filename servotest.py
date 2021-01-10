from board import SCL, SDA
import busio, time
from adafruit_pca9685 import PCA9685

# Servo Channels
IRIS_SERVO = 4
DOME_LIGHTS = 0
IRIS_LIGHT = 1

FREQUENCY = 50
PERIOD = 1.0 / float(FREQUENCY) * 1000.0
print(str(PERIOD))

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
    value = 0.2 + (value * 0.6) # normalise between 0.2 and 0.8
    value = 1.0 - value # reverse value
    value = value + 1.0 # change to range 1.2 to 1.8
    print("Value: " + str(value))
    duty_cycle = int(value / (PERIOD / 65535.0))
    print("Duty Cycle: " + str(duty_cycle))
    pca.channels[channel].duty_cycle = duty_cycle

def dalek_light(channel,value):
    pca.channels[channel].duty_cycle = int(value * 65535.0)

def status(direction):
    dalek_servo(IRIS_SERVO, 1-direction)
    dalek_light(IRIS_LIGHT, 1-direction)
    dalek_light(DOME_LIGHTS, 1-direction)
    for x in range(0,1000):
        if direction:
            value = (float(x) / 1000.0)**4
        else:
            value = (1.0 - (float(x) / 1000.0))**4
        dalek_servo(IRIS_SERVO, value)
        dalek_light(IRIS_LIGHT, value)
        dalek_light(DOME_LIGHTS, value)
        time.sleep(3.0/1000.0)
    dalek_servo(IRIS_SERVO, direction)
    dalek_light(IRIS_LIGHT, direction)
    dalek_light(DOME_LIGHTS, direction)

time.sleep(5.0)
status(True)
time.sleep(5.0)
status(False)
time.sleep(5.0)
status(True)
time.sleep(5.0)
status(False)

while True:
    value = float(input("Value 0.0 to 1.0 - "))
    dalek_servo(IRIS_SERVO,value)
    dalek_light(IRIS_LIGHT,value)
    dalek_light(DOME_LIGHTS,value)

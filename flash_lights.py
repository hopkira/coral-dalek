import pyaudio
from threading import Thread
import time
import numpy as np

# import servo board
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Vales to control whether dome lights are on or off
VOL_MIN = 80
VOL_MAX = 2250
RATE = 44100  # recording rate in Hz
MAX = 400  # minimum volume level for dome lights to illuminate
ON = 1.0
CHUNK = 2**13  # buffer size for audio capture and analysis

FREQUENCY = 50
PERIOD = 1.0 / float(FREQUENCY) * 1000.0

DOME_LIGHTS = 0

# create iris servo
i2c_bus = busio.I2C(SCL, SDA)
pca = PCA9685(i2c_bus)
pca.frequency = FREQUENCY

# Sets up a daemon thread to flash lights in line with sound

def dalek_light(channel,value):
    """
    Changes the level of illumination of a light attached to the
    PWM output of the servo controller.

    Args:
        channel (int): the channel number of the servo (range 0-16)
        value (float): value between 0.0 and 1.0
    """
    pca.channels[channel].duty_cycle = int(value * 65535.0)

def flash_dome_lights():
    ''' Daemon thread to flash lights based on microphone noise '''

    while True:
        try:
            data = np.frombuffer(stream.read(CHUNK, False),dtype=np.int16)
            vol = abs(int(np.average(np.abs(data))))
            print(vol)
            if vol > VOL_MIN:
                vol = vol - VOL_MIN
            else:
                vol = 0
            vol = vol * ON / VOL_MAX
            if vol > ON:
                vol =  ON
            dalek_light(DOME_LIGHTS, vol / ON)
        except ValueError:
            print ("Volume out of range: " + vol)

print("Starting audio thread...")
p = pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK, input_device_index=1)
domeLightsThread = Thread(target=flash_dome_lights, daemon=True)
domeLightsThread.start()
print("Audio thread started...")

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Lights stopped by user.")
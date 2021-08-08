import pyaudio
from threading import Thread

# Vales to control whether dome lights are on or off
VOL_MIN = 5000
VOL_MAX = 20000
RATE = 44100  # recording rate in Hz
MAX = 10000  # minimum volume level for dome lights to illuminate
ON = 1.0
CHUNK = 2**13  # buffer size for audio capture and analysis

# Sets up a daemon thread to flash lights in line with sound
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
        time.sleep(0.1)
except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Lights stopped by user.")
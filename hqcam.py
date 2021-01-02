import sys, time
import picamera
import picamera.array as npcam
with picamera.PiCamera(sensor_mode = 2) as camera:
    with npcam.PiArrayOutput(camera) as np_frame:
        camera.resolution = (2592, 1944)
        while True:
            try:
                start = time.perf_counter()
                camera.capture(np_frame, 'rgb')
                print(f'Captured {np_frame.array.shape[1]}x{np_frame.array.shape[0]} numpy array')
                np_frame.truncate(0)
                end = time.perf_counter()
                print(f'Frame taken in {(end-start)*1000} ms')
            except KeyError:
                camera.close()
                sys.exit("Numpy capture complete")
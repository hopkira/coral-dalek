import sys, time
import picamera
import picamera.array

with picamera.PiCamera(sensor_mode = 3) as camera:
        with picamera.array.PiRGBArray(camera) as np_frame:
            camera.resolution = (4056, 3040)
            while True:
                start = time.perf_counter()
                camera.capture(np_frame, 'rgb')
                print(f'Captured {np_frame.array.shape[1]}x{np_frame.array.shape[0]} numpy array')
                np_frame.truncate(0)
                end = time.perf_counter()
                print(f'Frame taken in {(end-start)*1000} ms')

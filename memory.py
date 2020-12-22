import sys, time
import picamera
with picamera.PiCamera(sensor_mode = 2) as camera:
    camera.resolution = (2592, 1944)
    while True:
        try:
            start = time.perf_counter()
            camera.capture('/var/tmp/capture.jpg', 'rgb')
            end = time.perf_counter()
            print(f'Frame taken in {(end-start)*1000} ms')
        except KeyError:
            camera.close()
            system.exit("Numpy capture complete")
import sys, time
import picamera
import numpy as np
import cv2
from PIL import Image
with picamera.PiCamera(sensor_mode = 2) as camera:
    camera.resolution = (2592, 1944)
    time.sleep(10)
    while True:
        try:
            start = time.perf_counter()
            camera.capture('/var/tmp/capture.jpg', 'rgb')
            np_img = cv2.imread('/var/tmp/capture.jpg', cv2.IMREAD_COLOR)
            # np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            print(f'Captured {np_img.shape[1]}x{np_img.shape[0]}x{np_img.shape[2]} numpy array')
            img = Image.fromarray(np_img)
            np_img.truncate(0)
            end = time.perf_counter()
            print(f'Frame taken in {(end-start)*1000} ms')
        except KeyError:
            camera.close()
            system.exit("Numpy capture complete")
import sys, time
import picamera
import numpy as np
import cv2
from PIL import Image
with picamera.PiCamera(sensor_mode = 2) as camera:
    camera.resolution = (2592, 1944)
    while True:
        try:
            start = time.perf_counter()
            image = np.empty((1944, 2592, 3), dtype=np.uint8)
            camera.capture(image, 'bgr')
            #camera.capture('/var/tmp/capture.jpg')
            #image = Image.open('/var/tmp/capture.jpg')
            #np_img = np.asarray(image)
            print(f'Captured {image.shape[1]}x{image.shape[0]}x{image.shape[2]} numpy array')
            end = time.perf_counter()
            print(f'Frame taken in {(end-start)*1000} ms')
        except KeyError:
            camera.close()
            sys.exit("Numpy capture complete")
import time

# import image and DL processing
import cv2
import numpy as np
import dlib
# from edgetpu.detection.engine import DetectionEngine
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

from imutils.video import VideoStream
from PIL import Image
from PIL import ImageDraw

HEIGHT = 1080 # pixels
WIDTH = 1920 # pixels
RESOLUTION = (WIDTH, HEIGHT)
FRAMERATE = 30

print("Loading face detection engine...")

interpreter = make_interpreter("./mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
interpreter.allocate_tensors()

print("Starting video stream...")
vs = VideoStream(src=0,
                 usePiCamera = False,
                 resolution=RESOLUTION,
                 framerate = FRAMERATE).start()

print("Waiting 5 seconds for camera feed to start...")
time.sleep(5.0) # wait for camera feed to start
print("Opening camera stream...")

while True:
    cam_frame = vs.read()
    np_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np_frame)
    
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    interpreter.invoke()

    face_box_list = detect.get_objects(interpreter, 0.7, scale)

    if len(face_box_list) > 0:
        for face in face_box_list:
            bbox = face.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='white')
    
    screen_image = image.convert('RGB')
    ImageDraw.Draw(screen_image)
    image.show()
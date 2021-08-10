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
from PIL import Image, ImageDraw

print("Loading face detection engine...")
interpreter = make_interpreter("/home/pi/coral-dalek/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
interpreter.allocate_tensors()

vc = cv2.VideoCapture(0)
if not vc.isOpened():
    print("Cannot open USB camera.")
    exit()

while True:
    ret, frame = vc.read()
    if not ret:
        print("No frame received from camera; exiting...")
        break
    image = Image.fromarray(frame)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    interpreter.invoke()
    face_box_list = detect.get_objects(interpreter, 0.7, scale)
    draw = ImageDraw.Draw(image)
    for face in face_box_list:
        bbox = face.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='white')
    displayImage = np.asarray(image)
    cv2.imshow('Object Detection', displayImage)
    if cv2.waitKey(1) == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()
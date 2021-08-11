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

# import local helper classes
from faceextractor import FaceDataExtractor
from recognizer import FaceRecognizer

print("Loading face detection engine...")
interpreter = make_interpreter("/home/pi/coral-dalek/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
interpreter.allocate_tensors()

print("Loading face landmark detection engine...")
shape_pred = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
face_ext = FaceDataExtractor()
print("Loading face recognition engine...")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")
face_recog = FaceRecognizer()

print("Starting video capture")

vc = cv2.VideoCapture(0)
if not vc.isOpened():
    print("Cannot open USB camera.")
    exit()

while True:
    ret, frame = vc.read()
    if not ret:
        print("No frame received from camera; exiting...")
        break
    # Convert frame from color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    interpreter.invoke()
    face_box_list = detect.get_objects(interpreter, 0.7, scale)
    draw = ImageDraw.Draw(image)
    for face in face_box_list:
        bbox = face.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='white')
        box = dlib.rectangle(left = bbox.xmin,
                             right = bbox.xmax,
                             top = bbox.ymin,
                             bottom = bbox.ymax)
        shape = shape_pred(frame, box)
        if shape:
            face_chip_img = dlib.get_face_chip(frame, shape)
            face_descriptor = facerec.compute_face_descriptor(face_chip_img)
            name = face_recog.recognize_face(face_descriptor, threshold = 0.9)
        if name:
            draw.text((bbox.xmin + 10, bbox.ymin + 10), name, fill='white')

    displayImage = np.asarray(image)
    displayImage = cv2.cvtColor(displayImage, cv2.COLOR_RGB2BGR)
    cv2.imshow('Object Detection', displayImage)
    if cv2.waitKey(1) == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()
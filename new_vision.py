import time

# import image and DL processing
import cv2
import numpy as np
import dlib
from random import randrange
# from edgetpu.detection.engine import DetectionEngine
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from scipy.interpolate import UnivariateSpline

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
print("Loading face recognitn engine...")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")
face_recog = FaceRecognizer()

output=False

# https://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html

if output:
    pov = 0
    overlay=[]
    overlay.append(cv2.imread('dalekpov-a.png'))
    overlay.append(cv2.imread('dalekpov-b.png'))
    overlay.append(cv2.imread('dalekpov-c.png'))

    def create_transform(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))

    inc_col = create_transform([0, 64, 128, 192, 256],[150, 175, 200, 225, 256])
    dec_col = create_transform([0, 64, 128, 192, 256],[28, 64, 90, 110, 128])

print("Starting video capture")

vc = cv2.VideoCapture(0)
if not vc.isOpened():
    print("Cannot open USB camera.")
    exit()

cap_width  = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap_fps = vc.get(cv2.CAP_PROP_FPS)
print(cap_width," x ", cap_height," @ ", cap_fps)

while True:
    try:
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
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='black')
            box = dlib.rectangle(left = bbox.xmin,
                                right = bbox.xmax,
                                top = bbox.ymin,
                                bottom = bbox.ymax)
            shape = shape_pred(frame, box)
            if shape:
                face_chip_img = dlib.get_face_chip(frame, shape)
                face_descriptor = facerec.compute_face_descriptor(face_chip_img)
                name = face_recog.recognize_face(face_descriptor, threshold = 0.55)
            if name:
                if output:
                    draw.text((bbox.xmin, bbox.ymin - 20), name, fill='black')
                else:
                    print(name)
        
        if output:
            displayImage = np.asarray(image)
            blue, green, red = cv2.split(displayImage)
            red = cv2.LUT(red, dec_col).astype(np.uint8)
            blue = cv2.LUT(blue, dec_col).astype(np.uint8)
            green = cv2.LUT(green, inc_col).astype(np.uint8)
            displayImage = cv2.merge((red, green, blue))

            # displayImage = cv2.cvtColor(displayImage, cv2.COLOR_BGR2GRAY)
            if (randrange(10) > 6): pov = randrange(3)
            displayImage = cv2.addWeighted(displayImage,0.8,overlay[pov],0.2,0)
            cv2.imshow('Dalek Eyestalk PoV', displayImage, cv2.WINDOW_NORMAL)
            if cv2.waitKey(1) == ord('q'):
                break
    except:
        vc.release()
        cv2.destroyAllWindows()
        sys.exit(0)
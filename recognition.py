from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import dlib
import imutils
import cv2
import time
import math
import sys

HEIGHT = 1088 # pixels
WIDTH = 1920 # pixels
RESOLUTION = (WIDTH, HEIGHT)

SENSOR_WIDTH = 6.3 # sensor width in mm
LENS_FOCAL_LENGTH = 6.0 # sensor focal length in mm
EYE_DISTANCE = 82.0 # distance between edges of eyes in mmm

PIX_TO_M = LENS_FOCAL_LENGTH * EYE_DISTANCE * float(WIDTH) / SENSOR_WIDTH / 1000.0

#white = dlib.rgb_pixel(255,255,255)

model = DetectionEngine("/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
shape_pred = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

#win = dlib.image_window()
#win.set_title("Homing Camera")

vs = VideoStream(src=0, usePiCamera = True, resolution=RESOLUTION, framerate = 30).start()

time.sleep(2.0) # wait for camera feed to start

def calc_position(eye_width, eye_offset):
    offset = pixel_to_m(eye_offset)
    dist = pixel_to_m(eye_width)
    return dict(angle = math.atan2(offset,dist), dist = dist)

def pixel_to_m(pixels):
    if pixels:
        return PIX_TO_M / pixels
    else:
        return 10.0

while True:
    try:
        cam_frame = vs.read()
        # frame = imutils.resize(frame, width=500)
        np_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        img_frame = Image.fromarray(np_frame)
        #win.set_image(np_frame)
        face_list = model.detect_with_image(img_frame,
            threshold = 0.7,
            keep_aspect_ratio = True, 
            relative_coord = False, 
            top_k = 1)
        for face in face_list:
            face_box = face.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = face_box
            box = dlib.rectangle(left=startX,right=endX,top=startY,bottom=endY)
            shape = shape_pred(np_frame, box)
            if shape:
                left_x = shape.part(0).x
                right_x = shape.part(3).x
                left_y = shape.part(0).y
                right_y = shape.part(3).y
                eye_width = ((((right_x - left_x )**2) + ((right_y - left_y)**2) )**0.5)
                eye_offset = (WIDTH / 2) - ((right_x + left_x) /2)
                position = calc_position(eye_width=eye_width, eye_offset=eye_offset)
                #print('Distance = %.2fpx Offset = %.2f px' % (eye_width, eye_offset))
                print('Distance = %.2fm Angle = %.2f radians' % (position['dist'], position['angle']))
                #win.clear_overlay()
                #win.add_overlay(shape,white)
                #win.add_overlay(box,white)
                # face_img = dlib.get_face_chip(np_frame, shape)
                # win.set_image(face_img)
                # face_descriptor = facerec.compute_face_descriptor(face_chip)
                # np.linalg.norm(known_faces - face, axis=1)
                # return np.linalg.norm(face_encodings - face_to_compare, axis=1)
                # dlib.full_object_detection, idx:
    except KeyboardInterrupt:
        vs.stop()
        print("Stopped video stream")
        sys.exit(0)
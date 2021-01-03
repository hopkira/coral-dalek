import dlib, imutils, cv2
import time, math, sys, pickle
# import datetime
# import numpy as np
from faceextractor import FaceDataExtractor
from recognizer import FaceRecognizer
from facelist import FaceList
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image

HEIGHT = 1080 # pixels
WIDTH = 1920 # pixels
RESOLUTION = (WIDTH, HEIGHT)
FRAMERATE = 30

SENSOR_WIDTH = 6.3 # sensor width in mm
LENS_FOCAL_LENGTH = 6.0 # sensor focal length in mm
EYE_DISTANCE = 82.0 # distance between edges of eyes in mmm (NOT pupil distance)

PIX_TO_M = LENS_FOCAL_LENGTH * EYE_DISTANCE * float(WIDTH) / SENSOR_WIDTH / 1000.0

print("Loading detection engine...")
model = DetectionEngine("/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
face_ext = FaceDataExtractor()
face_list = FaceList()

win = dlib.image_window()
win.set_title("Dalek Viewpoint")

def pix2metre(offset, eye_width):
    # returns offset from centre in m
    return offset / eye_width * EYE_DISTANCE / 1000

def calc_position(eye_width, eye_h_offset, eye_v_offset):
    # returns dictionary of position information
    z_dist = PIX_TO_M / eye_width
    h_offset = pix2metre(eye_width = eye_width, offset = eye_h_offset)
    v_offset = pix2metre(eye_width = eye_width, offset = eye_v_offset)
    return dict(h_offset = h_offset,
                v_offset = v_offset,
                h_angle = math.atan2(h_offset, z_dist),
                v_angle = math.atan2(v_offset, z_dist),
                z_dist = z_dist)

print("Starting video stream...")
vs = VideoStream(src=0, 
                 usePiCamera = True, 
                 resolution=RESOLUTION, 
                 framerate = FRAMERATE).start()

print("Waiting 5 seconds for camera feed to start...")
time.sleep(5.0) # wait for camera feed to start
print("Opening camera stream...")

while True:
    try:
        ##timestart = datetime.datetime.now()
        cam_frame = vs.read()
        # frame = imutils.resize(frame, width=500)
        np_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        img_frame = Image.fromarray(np_frame)
        win.set_image(np_frame)
        face_box_list = model.detect_with_image(img_frame,
            threshold = 0.9,
            keep_aspect_ratio = True, 
            relative_coord = False, 
            top_k = 1)
        face_list.expire_faces()
        for face_box in face_box_list:
            face_data = face_ext.extract_data(face = face_box, np_frame = np_frame)
            if face_data:
                # update face list with face_data
                face_data['position'] = calc_position(eye_width = face_data['eye_width'],
                                                      eye_h_offset = face_data['eye_h_offset'],
                                                      eye_v_offset = face_data['eye_v_offset'])
                face_list.update_list(face_data)

        for face in face_list.faces:
            print('%s is at %.2fm and a bearing of %.2f radians with age %.0f' % (face['name'], face['position']['z_dist'], face['position']['h_angle'], face['age']))
            startX = face['left_x']
            startY = face['left_y']
            endX = face['right_x']
            endY = face['right_y']
            rect = dlib.rectangle(startX, startY, endX, endY)
            win.add_overlay(rect)
            #cv2.rectangle(cam_frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            #y = startY - 15 if startY - 15 > 15 else startY + 15
            #text = face['name']
            #cv2.putText(cam_frame, text, (startX, y),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # win.set_image(face['face_chip_img'])
            #win.set_title("XXX")
            #timeend = datetime.datetime.now()
            # print(str(timeend-timestart))

    except KeyboardInterrupt:
        vs.stop()
        print("Stopped video stream")
        sys.exit(0)
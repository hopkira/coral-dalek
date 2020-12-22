import dlib, imutils, cv2
import time, math, sys, pickle
import numpy as np
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
from face_extraction import extract_face_data

HEIGHT = 1080 # pixels
WIDTH = 1920 # pixels
RESOLUTION = (WIDTH, HEIGHT)
FRAMERATE = 1

SENSOR_WIDTH = 6.3 # sensor width in mm
LENS_FOCAL_LENGTH = 6.0 # sensor focal length in mm
EYE_DISTANCE = 82.0 # distance between edges of eyes in mmm (NOT pupil distance)

PIX_TO_M = LENS_FOCAL_LENGTH * EYE_DISTANCE * float(WIDTH) / SENSOR_WIDTH / 1000.0

DESCRIPTORS = "face_descriptors.npy"
LABELS = "labels.pickle"

print("Loading detection engine...")
model = DetectionEngine("/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")

print("Retrieving recognition database...")
descriptors = np.load(DESCRIPTORS)
f = open(LABELS, 'rb')
labels = pickle.load(f) # in bytes

win = dlib.image_window()
win.set_title("Matched faces")

def calc_position(eye_width, eye_offset):
    dist = PIX_TO_M / eye_width
    offset = eye_offset / eye_width * EYE_DISTANCE / 1000
    return dict(angle = math.atan2(offset,dist), dist = dist)

def recognize_face(face_descriptor, threshold = 0.7):
    distances = np.linalg.norm(descriptors - face_descriptor, axis=1)
    argmin = np.argmin(distances)
    min_dist = distances[argmin]
    if min_dist > threshold:
        name = "Unknown"
    else:
        name = labels[argmin]
    return name

print("Starting video stream...")
vs = VideoStream(src=0, usePiCamera = True, resolution=RESOLUTION, framerate = FRAMERATE).start()

print("Waiting 5 seconds for camera feed to start...")
time.sleep(5.0) # wait for camera feed to start
print("Opening camera stream...")

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
            face_data = extract_face_data(face = face, np_frame = np_frame)
            if face_data:
                position = calc_position(eye_width = face_data['eye_width'], eye_offset = face_data['eye_offset'])
                name = recognize_face(face_descriptor = face_data['face_descriptor'])
                print('%s is at %.2fm and a bearing of %.2f radians' % (name, position['dist'], position['angle']))
                win.set_image(face_data['face_chip_img'])
    except KeyboardInterrupt:
        vs.stop()
        print("Stopped video stream")
        sys.exit(0)
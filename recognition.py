from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
import dlib
import imutils
import cv2
import time

height = 720
width = 1280
resolution = (width, height)

model = DetectionEngine("/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
sp = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

win = dlib.image_window()

vs = VideoStream(src=0, usePiCamera = True, resolution=resolution, framerate = 30).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame)
    np_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np_frame)
    win.clear_overlay()
    win.set_image(np_frame)
    results = model.detect_with_image(frame,
        threshold = 0.9,
        keep_aspect_ratio = True, 
        relative_coord = False, 
        top_k = 1)
    for r in results:
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
vs.stop()
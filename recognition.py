from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import dlib
import imutils
import cv2
import time

height = 720
width = 1080
resolution = (width, height)

model = DetectionEngine("/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
shape_pred = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

win = dlib.image_window()

vs = VideoStream(src=0, usePiCamera = True, resolution=resolution, framerate = 60).start()
time.sleep(2.0)

while True:
    try:
        cam_frame = vs.read()
        # frame = imutils.resize(frame, width=500)
        np_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        img_frame = Image.fromarray(np_frame)
        # win.set_image(np_frame)
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
            # win.clear_overlay()
            # win.add_overlay(d)
            # win.add_overlay(shape)
            face_img = dlib.get_face_chip(np_frame, shape)
            win.set_image(face_img)
            #face_descriptor = facerec.compute_face_descriptor(face_chip)
    except KeyboardInterrupt:
        vs.stop()
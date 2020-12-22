# This routine iterates across all files and directories
# in the training directory.
# Each person has their own directory, with the photos being
# stored as .png files.
# The files should be called '0.png', '1.png' etc.

import os, sys, pickle
import cv2, dlib, time
import numpy as np
from face_extraction import extract_face_data
from PIL import Image
from edgetpu.detection.engine import DetectionEngine

SAMPLES = 8
CONFIDENCE = 0.7

model = DetectionEngine("/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
DESCRIPTORS = "face_descriptors"
LABELS = "labels.pickle"

win = dlib.image_window()
win.set_title("Training faces")

def save_descriptor(descriptor, label):
    initialize = False
    try:
        # deserialize descriptors and labels from disk
        descriptors = np.load(DESCRIPTORS)
        f = open(LABELS, 'rb')
        labels = pickle.load(f) # in bytes
    except IOError:
        initialize = True # files do not exist
    if initialize:
        # initialize with calling parameters
        descriptors = descriptor
        labels = [label]
    else:
        # add calling parameters to end of existing lists
        descriptors = np.concatenate([descriptors, descriptor], axis=0)
        labels.append(label)
    # Serialize descriptors and labels
    np.save(DESCRIPTORS, descriptors)
    with open(LABELS, "wb") as f:
        pickle.dump(labels, f)
    return True

for root, dirs, files in os.walk('/home/pi/dalek-doorman/training'):
    for dir in dirs:
        print('Training subject: ' + dir)  # announce who is being trained
        # iterate over this number of photo samples for each person
        for num in range(0, SAMPLES):
            # create a fully described path for each training image
            file_name = str(num)+'.png'
            train_filename = (os.path.join(root, dir, file_name))
            print('Training ' + dir + ' with ' + train_filename)
            np_img = cv2.imread(train_filename, cv2.IMREAD_COLOR)
            img = Image.fromarray(np_img)
            face_list = model.detect_with_image(img,
                threshold=0.7,
                keep_aspect_ratio=True, 
                relative_coord=False, 
                top_k=1)
            if len(face_list) < 1:
                sys.exit("Face not found in training image")
            if len(face_list) > 1:
                raise ValueError("More than one face found in training image")
            face = face_list[0]
            face_data = extract_face_data(face = face, np_frame = np_img)
            if face_data:
                win.set_title(file_name)
                win.set_image(face_data['face_chip_img'])
                time.sleep(5)
                save_descriptor(descriptor = face_data['face_descriptor'], label = dir)
sys.exit("Training completed successfully")
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
DESCRIPTORS = "./face_descriptors"
LABELS = "./labels.pickle"

win = dlib.image_window()
win.set_title("Training faces")

initialize = False

def save_descriptor(descriptor, label):

    return True

for root, dirs, files in os.walk('/home/pi/dalek-doorman/training'):
    for file_name in files:
        # create a fully described path for each training image
        # file_name = str(num)+'.png'
        train_filename = os.path.join(root,file_name)
        directory = root.split(os.path.sep)[-1]
        np_img = cv2.imread(train_filename, cv2.IMREAD_COLOR)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np_img)
        win.set_image(np_img)
        win.set_title(directory)
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
            win.set_image(face_data['face_chip_img'])
            print("Got to saving bit")
            try:
                # deserialize descriptors and labels from disk
                descriptors = np.load(DESCRIPTORS)
                f = open(LABELS, 'rb')
                labels = pickle.load(f) # in bytes
                print(str(len(labels)))
            except IOError as e:
                print(e)
                initialize = True # files do not exist
                print("Files do not exist")
            if initialize:
                # initialize with calling parameters
                print("Creating new files")
                descriptors = face_data['face_descriptor']
                labels = [directory]
            else:
                # add calling parameters to end of existing lists
                descriptors = np.concatenate([descriptors, face_data['face_descriptor']], axis=0)
                labels.append(directory)
            # Serialize descriptors and labels
            np.save(DESCRIPTORS, descriptors)
            with open(LABELS, "wb") as f:
                pickle.dump(labels, f)
            print("Loaded " + str(train_filename) + " under " + str(directory))
sys.exit("Training completed successfully")
#
# Licensed under the The Unlicense
#
"""Enrolls face images into a recognition database.  Database includes a label
for each face and a 128D encoding created by dlib

python3 training.py \
-v \
--model /usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite \
--label ./labels.pickle \
--descriptor ./face_descriptors.npy \
--input /home/pi/dalek-doorman/training

"""

import argparse
import os
import sys
import time
import pickle
import cv2
import dlib
import numpy as np
from PIL import Image
from edgetpu.detection.engine import DetectionEngine
from faceextractor import FaceDataExtractor

parser = argparse.ArgumentParser()
parser.add_argument('-v',
    help = "Preview encoded images",
    action = 'store_true')
parser.add_argument('--model',
    help='Full path to mobilenet tflite model',
    default = "/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
parser.add_argument('--label',
    help='Label file path.',
    default = "./labels.pickle")
parser.add_argument('--descriptor',
    help='Descriptor file path.',
    default = "./face_descriptors.npy")
parser.add_argument('--input',
    help='Training image path.',
    default = "/home/pi/dalek-doorman/training")

args = parser.parse_args()

model = DetectionEngine(args.model)
face_ext = FaceDataExtractor()
DESCRIPTORS = args.descriptor
LABELS = args.label
initialize = False

if args.v:
    win = dlib.image_window()

for root, dirs, files in os.walk(args.input):
    for file_name in files:
        # create a fully described path for each training image
        # file_name = str(num)+'.png'
        train_filename = os.path.join(root,file_name)
        directory = root.split(os.path.sep)[-1]
        print(train_filename)
        np_img = cv2.imread(train_filename, cv2.IMREAD_COLOR)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
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
        face_data = face_ext.extract_data(face = face, np_frame = np_img)
        if face_data:
            if args.v:
                win.set_title(directory)
                win.set_image(face_data['face_chip_img'])
                time.sleep(5.0)
            try:
                # deserialize descriptors and labels from disk
                descriptors = np.load(DESCRIPTORS)
                f = open(LABELS, 'rb')
                labels = pickle.load(f)
            except IOError as error:
                print("{error} - Recognition DB not found")
                initialize = True # files do not exist
            if initialize:
                print("Creating new recognition DB")
                descriptors = face_data['face_descriptor']
                labels = [directory]
                initialize = False
            else:
                # add calling parameters to end of existing lists
                descriptors = np.concatenate([descriptors, face_data['face_descriptor']], axis=0)
                labels.append(directory)
            # Serialize descriptors and labels
            np.save(DESCRIPTORS, descriptors)
            with open(LABELS, "wb") as f:
                pickle.dump(labels, f)
            print(f"Loaded record #{len(labels)} {train_filename} as {directory}")
sys.exit("Training completed successfully")

#
# Licensed under the The Unlicense
#
"""Enrolls face images into a recognition database.  Database includes a label
for each face and a 128D encoding created by dlib

python3 training.py \
--model /usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite \
--label ./labels.pickle \
--descriptor ./face_descriptors.npy \
--input /home/pi/dalek-doorman/training

"""

import argparse
import os, sys, pickle
import cv2, dlib, time
import numpy as np
from face_extraction import extract_face_data
from PIL import Image
from edgetpu.detection.engine import DetectionEngine

SAMPLES = 8
CONFIDENCE = 0.7


win = dlib.image_window()
win.set_title("Training faces")

initialize = False

parser = argparse.ArgumentParser()
parser.add_argument('--model',
    help='Full path to mobilenet tflite model',
    nargs = '?',
    const = "/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
parser.add_argument('--label', 
    help='Label file path.',
    nargs = '?',
    const = "./labels.pickle")
parser.add_argument('--descriptor', 
    help='Descriptor file path.', 
    nargs = '?',
    const = "./face_descriptors.npy")
parser.add_argument('--input',
    help='Training image path.',
    nargs = '?',
    const = "/home/pi/dalek-doorman/training")

args = parser.parse_args()

model = DetectionEngine(args.model)
DESCRIPTORS = args.descriptor
LABELS = args.label

def save_descriptor(descriptor, label):

    return True

for root, dirs, files in os.walk(args.input):
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
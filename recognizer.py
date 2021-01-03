import numpy as np
import pickle

class FaceRecognizer():

    DESCRIPTORS = "face_descriptors.npy"
    LABELS = "labels.pickle"

    def __init__(self):
        print("Retrieving recognition database...")
        self.descriptors = np.load(FaceRecognizer.DESCRIPTORS)
        # will be loaded as a 1D array, so needs to be
        # reshaped back into a n x 128 arrary
        self.descriptors = self.descriptors.reshape (-1,128)
        f = open(FaceRecognizer.LABELS, 'rb')
        self.labels = pickle.load(f) # in bytes
        
    def recognize_face(self, face_descriptor, threshold = 0.7):
        distances = np.linalg.norm(self.descriptors - face_descriptor, axis=1)
        argmin = np.argmin(distances)
        min_dist = distances[argmin]
        if min_dist > threshold:
            name = "Unknown"
        else:
            name = self.labels[argmin]
            print("Identified " + str(name))
        return name
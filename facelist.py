from recognizer import FaceRecognizer

class FaceList:
    """
    List of active faces from face recognition
    """

    EXPIRY = 10
    MAX_DIST = 1.0

    def __init__(self):
        self.faces = []
        self.face_recog = FaceRecognizer()

    def find_face_dist(self, pos1, pos2):
        # takes two positions and returns euclidean distance
        y_dist = pos2['h_offset'] - pos1['h_offset']
        x_dist = pos2['v_offset'] - pos1['v_offset']
        z_dist = pos2['dist'] - pos1['dist']
        dist = ((x_dist**2) + (y_dist**2) + (z_dist**2)) ** 0.5
        return dist
    
    def expire_faces(self):
        for face in self.faces:
            face['age'] += 1
            if face['age'] > 100:
                self.faces.remove(face)

    def find_close_face(self, search_face):
        for face in self.faces:
            # cycle through faces and calculate Euclidean distance
            # and if within MAX_DIST then call it the same face
            dist = self.find_face_dist(search_face['position'], face['position'])
            if dist > FaceList.MAX_DIST:
                self.add_face(search_face)
            else:
                return None

    def add_face(self, face_dict):
        # identify face and update face_dict
        # with name etc. 
        face_dict['age'] = 0
        face_dict['name'] = self.face_recog.recognize_face(face_dict['face_descriptor'])
        self.faces.append(face_dict)

    def update_list(self, face_dict):
        if len(self.faces) == 0:
            self.add_face(face_dict)
        else:
            self.find_close_face(face_dict)
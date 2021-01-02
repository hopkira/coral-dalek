import recognition

class FaceList:
    """
    List of active faces from face recognition
    """

    EXPIRY = 10
    MAX_DIST = 1.0

    def __init__(self):
        self.faces = []
    
    def expire_faces(self):
        for face in self.faces:
            face.age += 1
            if face.age > 10:
                self.faces.remove(face)

    def find_close_face(self, face_dict):
        face = 0
        return face

    def add_face(self, face_dict):
        # identify face
        face_dict.age = 0

    def update_list(self, face_dict):
        if len(self.faces) == 0:
            self.add_face(face_dict)
        nearest_neighbour = self.find_close_face(face_dict)
        if nearest_neighbour < FaceList.MAX_DIST:
            self.faces.append(face_dict)
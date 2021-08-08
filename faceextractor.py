import dlib
# from edgetpu.detection.engine import DetectionEngine

class FaceDataExtractor:

    def __init__(self):
        self.shape_pred = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

    def extract_data(self, np_frame, face):
        width = np_frame.shape[1]
        height = np_frame.shape[0]
        face_box = face.BBox.flatten().astype("int") # changed to BBox object
        (startX, startY, endX, endY) = face_box
        box = dlib.rectangle(left = startX,
                            right = endX,
                            top = startY,
                            bottom = endY)
        shape = self.shape_pred(np_frame, box)
        if shape:
            face_chip_img = dlib.get_face_chip(np_frame, shape)
            # win.set_image(face_img)
            face_descriptor = self.facerec.compute_face_descriptor(face_chip_img)
            # np.linalg.norm(known_faces - face, axis=1)
            # return np.linalg.norm(face_encodings - face_to_compare, axis=1)
            # dlib.full_object_detection, idx:
            # 0 is the 
            left_x = shape.part(0).x
            right_x = shape.part(2).x
            left_y = shape.part(0).y
            right_y = shape.part(2).y
            eye_width = ((((right_x - left_x )**2) + ((right_y - left_y)**2) )**0.5)
            eye_h_offset = ((right_x + left_x) /2) - (width / 2)
            eye_v_offset = ((right_y + left_y) /2) - (height / 2)
            return dict(eye_width=eye_width, 
                        eye_h_offset = eye_h_offset,
                        eye_v_offset = eye_v_offset, 
                        face_descriptor = face_descriptor,
                        face_chip_img = face_chip_img,
                        left_x = left_x,
                        right_x = right_x,
                        left_y = left_y,
                        right_y = right_y)
        else:
            return None
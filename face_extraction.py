import dlib
from edgetpu.detection.engine import DetectionEngine

shape_pred = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

def extract_face_data(np_frame,face):
    width = np_frame.shape[0]
    print(np_frame.shape)
    face_box = face.bounding_box.flatten().astype("int")
    (startX, startY, endX, endY) = face_box
    box = dlib.rectangle(left=startX,right=endX,top=startY,bottom=endY)
    shape = shape_pred(np_frame, box)
    if shape:
        face_chip_img = dlib.get_face_chip(np_frame, shape)
        # win.set_image(face_img)
        face_descriptor = facerec.compute_face_descriptor(face_chip_img)
        # np.linalg.norm(known_faces - face, axis=1)
        # return np.linalg.norm(face_encodings - face_to_compare, axis=1)
        # dlib.full_object_detection, idx:
        left_x = shape.part(0).x
        right_x = shape.part(3).x
        left_y = shape.part(0).y
        right_y = shape.part(3).y
        eye_width = ((((right_x - left_x )**2) + ((right_y - left_y)**2) )**0.5)
        eye_offset = ((right_x + left_x) /2) - (width / 2)
        return dict(eye_width=eye_width, 
                    eye_offset = eye_offset, 
                    face_descriptor = face_descriptor,
                    face_chip_img = face_chip_img)
    else:
        return None
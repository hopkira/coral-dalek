# USAGE
    # python3 find_face.py --model /usr/share/edgetpu/examples/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import argparse
import imutils
import time
import cv2
import dlib

height = 720
width = 1280
resolution = (width, height)

face_px = 112

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0, usePiCamera = True, resolution=resolution, framerate = 30).start()
#vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	frame = vs.read()
	# frame = imutils.resize(frame, width=500)
	frame = imutils.resize(frame)
	orig = frame.copy()

	# prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = Image.fromarray(frame)

	# make predictions on the input frame
	start = time.time()
	results = model.detect_with_image(frame, threshold=args["confidence"],
		keep_aspect_ratio=True, relative_coord=False, top_k=1)
	end = time.time()

	# loop over the results
	for r in results:
		# extract the bounding box and box and predicted class label
		box = r.bounding_box.flatten().astype("int")
		(startX, startY, endX, endY) = box
		# label = labels[r.label_id]
		x_dim = endX - startX
		y_dim = endY - startY

		if x_dim > face_px and y_dim >= face_px:
			roi = orig[startY:endY, startX:endX]
			# aspect_ratio = x_dim/y_dim
			if x_dim >= y_dim :
				scale = y_dim / face_px
				x_size = int(x_dim / scale)
				x_start = int((x_size - face_px) / 2)
				roi = cv2.resize(roi, (x_size, face_px))
				roi = roi[0:111, x_start:x_start + face_px - 1]		
			else:
				scale = x_dim / face_px
				y_size = int(y_dim/scale)
				y_start = int((y_size - face_px) /2)
				roi = cv2.resize(roi, (face_px, y_size))	
				roi = roi[y_start:y_start + face_px - 1, 0:111]	
			cv2.imshow("Face",roi)

		# draw the bounding box and label on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		text = "{}: {:.2f}%".format("face", r.score * 100)
		cv2.putText(orig, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# show the output frame and wait for a key press
	# cv2.imshow("Dalek Viewpoint", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

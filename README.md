# Dalek Doorman with Coral TPU
In this enhancement to the original Dalek Doorman, a Coral TPU is used to speed up face detection within the video camera image so that family can be recognized and welcomed and intruders can be exterminated.  The current version of the code uses:
* Raspberry Pi (3 or beyond recommended)
  * Convincing dalek speech generated directly in Pi
  * External amplifier and speakers recommended (via 3.5mm audio out)
* Google Coral USB TPU (8-bit)
  * Real-time face detection
* PCA9685 16 channel, 12-bit PWM servo controller to drive the iris servo and control the dome and iris lights
  * Opens and closes the eyestalk iris
  * Flashes the dome lights
  * Gradually lights and dims the iris lights
* USB microphone (to detect when to flash the dome lights)
  * Flashes dome lights in time with Dalek speech
* USB webcam or Pi Camera
  * With the ability to generate and show the Dalek pov with adjusted colours and animated overlays

The following assets in this repository are used by the current configuration.

|filename|file used to|
|---|---|
|requirements.txt|Defines the software used by the dalek; it is recommended you create a python virtual environment and then install ```using pip install -r requirements.txt```|
|new_coral_dalek.py|The main program  ```python new_coral_dalek.py -h``` will explain command line options, but basically a finite state machine that controls the behaviour of the dalek|
|mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite|Face detection neural net model that is compiled and runs on the Coral USB TPU - this allows the Dalek to very quickly react when it sees a face|
|shape_predictor_5_face_landmarks.dat|Another neural net model.  This one looks at the faces detected above and identifies the corners of the eyes and bottom of the nose. This information is then used to normalize the image (e.g. rotating it) to improve the accuracy of the face recognition engine.
|dalek-pov-a.png|Three graphics files (a, b and c) that are used to generate the changing symbols and the cross hair in the dalek point of view video stream
|training.py| This program runs against a selection of labelled images to generate a 128 dimensional vector for each image and then stores thise vectors and labels on the Pi filestore.  This training dataset is persisted in the Pi filesystem as numpy facedescriptors and corresponding name labels.  
|facedescriptors.npy|A list of 128-dimension vectors describing each face extracted from the training photos|
labels.pickle|The corresponding list of name labels for each trained photo|
|recognizer.py|This is the python module that recognizes people from the detected faces. A vector is generated for each facebox supplied and then the distance between the vectors already inthe database is used to determine if the face is a known or not, and if it is known, who is the people who are currently visible to the dalek. If there is no vector nearby (within a tolerance that can be changed at program start) then the person is labelled as unknown.|

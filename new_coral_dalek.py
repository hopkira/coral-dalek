#!/usr/bin/env python3
"""Dalek State Machine

This module provides a state machine that enables a Dalek to recognise people within
its field of view and either welcome them by name, or threaten them if it doesn't
recognise them.  It assumes:

    * the use of an Adafruit Servo Controller to control an iris servo
      and the lights within the eye and dome (using a TIP120 transistor to amplify the PWM signal)
    * a Pi High Quality camera with 6mm lens
    * a Google Coral TPU to provide acceleration for the face detection function
    * a database of face descriptors and labels as created by the training.py program
      (so be sure to run that program first!)
    * a USB microphone and an amplifier for the dalek voice

The Dalek operator can also optionally activate and deactivate the Dalek via an MQTT message
over Bluetooth BLE; this assumes that there is a localhost MQTT server

Example:
    $ python3 dalek_state.py

Todo:
    * Add in the hover lights on an additional servo channel
    * Use the location and eye distance of the detected face to calculate distance and
      bearing so the dome can swivel to look directly at the persons being talked to

Userful blog posts for setup: https://k9-build.blogspot.com/search/label/dalek

Dalek and K9 word marks and logos are trade marks of the British Broadcasting Corporation and
are copyright BBC/Terry Nation 1963

"""
import os
import time
import random
from threading import Thread

# import servo board
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# import image and DL processing
import cv2
import numpy as np
import dlib
# from edgetpu.detection.engine import DetectionEngine
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

from imutils.video import VideoStream
from PIL import Image

# import audio and messaging
import pyaudio
import paho.mqtt.client as mqtt

# import local helper classes
from faceextractor import FaceDataExtractor
from recognizer import FaceRecognizer


FREQUENCY = 50
PERIOD = 1.0 / float(FREQUENCY) * 1000.0

# create iris servo
i2c_bus = busio.I2C(SCL, SDA)
pca = PCA9685(i2c_bus)
pca.frequency = FREQUENCY

# Initialise the pygame mixer for sound and sound effect
#pygame.mixer.init()
#pygame.mixer.music.load("./controlroom.wav")

DEAD_TIME = 30  # minimum time in seconds between doorman annoucemnents
EVENT_GAP = 5  # maximum time window in seconds for valid detection events
# no. of recognition events needed with less than
# EVENT_GAP between them to hit threshold
THRESHOLD = 3
UNKNOWN_THRESHOLD = 5  # numer of unknown events to hit threshold
UNKNOWN_GAP = 30  # maximum time window in seconds for valid uknown events
SAMPLES = 8  # number of training photos per person (limit 50 in total)
CHUNK = 2**13  # buffer size for audio capture and analysis
RATE = 44100  # recording rate in Hz
MAX = 10000  # minimum volume level for dome lights to illuminate

# These control the three different dalek voices
SPEED_DEFAULT = 175
SPEED_DOWN = 125
AMP_UP = 200
AMP_DEFAULT = 190
AMP_DOWN = 180
PITCH_DEFAULT = 99
PITCH_DOWN = 69
SOX_VOL_UP = 5000
SOX_VOL_DEFAULT = 20
SOX_VOL_DOWN = 10
SOX_PITCH_UP = 50
SOX_PITCH_DEFAULT = 0
SOX_PITCH_DOWN = -25

# Servo Channels
IRIS_SERVO = 4
DOME_LIGHTS = 0
IRIS_LIGHT = 1

# Convenience Servo Values
ON = 1.0
AWAKE = True
ASLEEP = False
OFF = 0.0
STEPS = 100
DURATION = 1.0
SERVO_MAX = 0.8
SERVO_MIN = 0.2

# Vales to control whether dome lights are on or off
VOL_MIN = 5000
VOL_MAX = 20000

HEIGHT = 1080 # pixels
WIDTH = 1920 # pixels
RESOLUTION = (WIDTH, HEIGHT)
FRAMERATE = 30

unknown_count = 0  # number of times an unknown face has been seen
unknown_seen = round(time.time())

print("Loading face detection engine...")

interpreter = make_interpreter(“/home/pi/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite”)
interpreter.allocate_tensors()
#model = DetectionEngine("/usr/share/edgetpu/examples/models/"
#                        "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
print("Loading face landmark detection engine...")
shape_pred = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
face_ext = FaceDataExtractor()
print("Loading face recognition engine...")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")
face_recog = FaceRecognizer()
print("Starting video stream...")
vs = VideoStream(src=0,
                 usePiCamera = False,
                 resolution=RESOLUTION,
                 framerate = FRAMERATE).start()

print("Waiting 5 seconds for camera feed to start...")
time.sleep(5.0) # wait for camera feed to start
print("Opening camera stream...")

def dalek_status(direction):
    """
    Opens or closes the dalek eye and lights

    Arg:
        direction (boolean): True to open, False to close
    """
    dalek_servo(IRIS_SERVO, 1-direction)
    dalek_light(IRIS_LIGHT, 1-direction)
    for pos in range(0, STEPS):
        if direction:
            value = (float(pos) / float(STEPS))**4
        else:
            value = (1.0 - (float(pos) / float(STEPS)))**4
        dalek_servo(IRIS_SERVO, value)
        dalek_light(IRIS_LIGHT, value)
        time.sleep(DURATION/STEPS)
    dalek_servo(IRIS_SERVO, direction)
    dalek_light(IRIS_LIGHT, direction)

def dalek_servo(channel,value):
    """
    Changes the servo position of a servo on the Adafruit controller.
    Maximum and minumum safe positions are pre-calculated.

    Args:
        channel (int): the channel number of the servo (range 0-16)
        value (float): value between 0.0 and 1.0
    """
    value = SERVO_MIN + (value * (SERVO_MAX - SERVO_MIN)) # normalise between MAX and MIN
    value = 1.0 - value # reverse value
    value = value + 1.0 # change to range 1.2 to 1.8
    duty_cycle = int(value / (PERIOD / 65535.0))
    pca.channels[channel].duty_cycle = duty_cycle

def dalek_light(channel,value):
    """
    Changes the level of illumination of a light attached to the
    PWM output of the servo controller.

    Args:
        channel (int): the channel number of the servo (range 0-16)
        value (float): value between 0.0 and 1.0
    """
    pca.channels[channel].duty_cycle = int(value * 65535.0)


class Person:
    '''The Person class represents the people known to the Dalek'''

    def __init__(self, name):
        '''The attributes are mostly about when the Dalek last saw them

        Attributes
        ----------
        name : str
            the name of the person
        detected: int
            time of last detection event
        detection_events: int
            number of detection events within EVENT_GAP
        last_seen: int
            last time Dalek greeted that person

        Methods
        -------

        just_seen :
            records a sighting of the person by the robot
        '''
        self.name = name
        self.detection_events = 0  #  number of detection events at init is zero
        self.detected = 0  #  time of last know detection event
        self.last_seen = 0  #  time of last announcement
        self.now = 0
        self.duration = 0
        self.gap = 0

    def just_seen(self):
        '''Record sighting of person'''

        self.now = round(time.time())  # record the time of the detection event
        self.duration = self.now - self.last_seen # work out how long since last greeting
        print("Just seen " + str(self.name) + " after " + str(self.duration) + "s")
        if self.duration > DEAD_TIME:  # tests if an announcment is allowed
            self.gap = self.now - self.detected  # gap = how long since last sighting
            self.detected = self.now  # record the time of the sighting
            self.detection_events += 1  # increment the sightings counter
            print("Seen " + str(self.name) +
                  " " + str(self.detection_events) +
                  " times.  Last time " + str(self.gap) +
                  "s ago")
            if self.gap < EVENT_GAP:  # is the gap shorter than the allowed gap?
                if self.detection_events >= THRESHOLD:  # has the threshold been met?
                    print("I have seen " + self.name +
                          " too many times for it to be a false postive.")
                    # as we are outside the dead time and the threshold has
                    # been met, then we make an annoucement by
                    # upadating the Cloudant db with the current time,
                    # resetting the detection events counter to zero and
                    # initiating the dalek greeting
                    self.last_seen = self.now
                    self.detection_events = 0
                    dalek_greeting(self.name)
                    dalek.on_event("greet")
                else:
                    print("Back to watching, detection events for " +
                          str(self.name) +
                          " stands at " +
                          str(self.detection_events))
                    return
            else:
                # as the event is outside the window, but a sighting
                # has happened then reset the counter to 1
                self.detection_events = 1
                print("Reset counter. Detection events for " +
                      str(self.name) +
                      " is set to " +
                      str(self.detection_events))
                return
        else:
            print("I've seen " + str(self.name) + ", but recently shouted at them.")
            return


class State(object):
    '''
    State parent class to support standard Python functions
    '''

    def __init__(self):
        print('Entering state:', str(self))

    def on_event(self, event):
        '''
        Incoming events processing is delegated to the child State
        to define and enable the valid state transitions.
        '''

    def run(self):
        '''
        Enable the state to do something - this is usually delegated
        to the child States)
        '''
        print('Run event for ' + str(self) + ' state not implemented')

    def __repr__(self):
        '''
        Leverages the __str__ method to describe the State.
        '''
        return self.__str__()

    def __str__(self):
        '''
        Returns the name of the State.
        '''
        return self.__class__.__name__


# Start Dalek states
class Waiting(State):

    '''
    The child state where the Dalek is scanning for faces, but appears dormant
    '''
    def __init__(self):
        super(Waiting, self).__init__()

    def run(self):
        faces = detect_faces()
        if len(faces) > 0:
            dalek.on_event('face_detected')

    def on_event(self, event):
        if event == 'silent':
            return Silent()
        if event == 'face_detected':
            return WakingUp()
        return self


class Silent(State):
    '''
    The child state where the Dalek does not react without a new signal
    from the Bangle.js watch
    '''

    def __init__(self):
        super(Silent, self).__init__()

    def run(self):
        time.sleep(0.1)

    def on_event(self, event):
        if event == 'waiting':
            return Waiting()
        return self


class WakingUp(State):
    '''
    The child state where the Dalek wakes up by turning its lights on and
    openning its Iris
    '''

    def __init__(self):
        super(WakingUp, self).__init__()
        dalek_status(AWAKE)

    def run(self):
        dalek.on_event('dalek_awake')

    def on_event(self, event):
        if event == 'dalek_awake':
            return Awake()
        return self


class Awake(State):
    '''
    The child state where the Dalek searches for a recognizable face
    '''

    def __init__(self):
        super(Awake, self).__init__()
        self.now = round(time.time())

    def run(self):
        countdown = DEAD_TIME + self.now - round(time.time())
        if countdown <= 0:
            dalek.on_event('timeout')
        else:
            print("Countdown timer:" + str(countdown))
        face_names = recognise_faces()
        if len(face_names) > 0:
            self.now = round(time.time())
            for name in face_names:
                if name == "Unknown":
                    dalek.on_event("exterminate")
                else:
                    dalek_greeting(name)
            dalek.on_event("greet")

    def on_event(self, event):
        if event == 'timeout':
            return FallingAsleep()
        if event == 'greet':
            return Greeting()
        if event == 'exterminate':
            return Exterminating()
        return self


class Greeting(State):
    '''
    The child state where the Dalek says goodbye to a known person
    '''

    def run(self):
        dalek.on_event('greet_done')

    def on_event(self, event):
        if event == 'greet_done':
            return Awake()
        return self


class Exterminating(State):
    '''
    The child state where the Dalek exterminates someone it doesn't know
    '''

    def __init__(self):
        super(Exterminating, self).__init__()
        self.now = round(time.time())
        self.unknown_count = 0

    def run(self):
        countdown = DEAD_TIME + self.now - round(time.time())
        if countdown <= 0:
            dalek.on_event('timeout')
        else:
            print("Countdown: " + str(countdown))
        face_names = recognise_faces()
        if len(face_names) > 0:
            self.now = round(time.time())
            for face in face_names:
                if face == "Unknown":
                    self.unknown_count += 1
                else:
                    self.unknown_count = 0
                    dalek.on_event("known_face")
        if self.unknown_count < UNKNOWN_THRESHOLD:
            print("Exterminating: unknown count - " + str(unknown_count))
        else:
            warning = ("You are|>unrecognized. Do not|>move!",
                       ">Halt|You are an|>enemy|of the|<Darleks.",
                       "You are|>unknown|<You will|be|>exterminated!",
                       "Intruder|>alert!",
                       "<Enemy|detected!|>Exterminate!",
                       "Halt. Do not|<move.|You will|>obey!",
                       "Obey the Darleks!|>Obey the Darleks!",
                       "Unknown human|<in hall|>Exterminate!",
                       "Do not|>move.|You will be|>exterminated!",
                       "Warning|>Warning|Do not move!")
            response = random_msg(warning)
            self.unknown_count = 0
            dalek_speak(response)
            dalek.on_event('death')

    def on_event(self, event):
        if event == 'death':
            return Awake()
        if event == 'timeout':
            return Awake()
        if event == 'known_face':
            return Awake()
        return self


class FallingAsleep(State):
    '''
    The child state where the Dalek returns to dormant state
    '''

    def __init__(self):
        super(FallingAsleep, self).__init__()
        dalek_status(ASLEEP)

    def run(self):
        dalek.on_event('asleep')

    def on_event(self, event):
        if event == 'asleep':
            return Waiting()
        return self

# End Dalek states.


class Dalek(object):
    '''
    A Dalek finite state machine that starts in waiting state and
    will transition to a new state on when a transition event occurs.
    It also supports a run command to enable each state to have its
    own specific behaviours
    '''

    def __init__(self):
        ''' Initialise the Dalek in its Waiting state. '''

        # Start with a default state.
        dalek_status(AWAKE)
        dalek_speak("I am Darlek Fry!")
        dalek_status(ASLEEP)
        self.state = Waiting()

    def run(self):
        ''' State behavior is delegated to the current state'''

        self.state.run()

    def on_event(self, event):
        '''
        Incoming events are delegated to the current state, which then
        returns the next valid state.
        '''

        # The next state will be the result of the on_event function.
        self.state = self.state.on_event(event)


def detect_faces():
    '''
    Takes a video frame and detects whether there is a face in the picture
    using the Coral TPU.

    This is much quicker than identifying the face, so it used to wake up
    the dalek.  This makes the recognition seem much more immediate.
    '''
    cam_frame = vs.read()
    np_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    img_frame = Image.fromarray(np_frame)
    
    image = Image.open(“image filename”)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    interpreter.invoke()

    face_box_list = detect.get_objects(interpreter, 0.9, scale)

    #face_box_list = model.detect_with_image(img_frame,
    #    threshold = 0.9,
    #    keep_aspect_ratio = True,
    #    relative_coord = False,
    #    top_k = 3)

    return face_box_list


def recognise_faces():
    '''
    Grabs a video frame and detects whether there are faces in the video image
    if there are, it attempts to identify them, returning a list of names, or
    unknown if someone unknown is in the image
    '''
    face_box_list = detect_faces()
    for face_box in face_box_list:
        face_data = face_ext.extract_data(face = face_box, np_frame = np_frame)
        if face_data:
            face_box = face_box.BBox.flatten().astype("int") # change BBox
            (start_x, start_y, end_x, end_y) = face_box
            box = dlib.rectangle(left = start_x,
                                right = end_x,
                                top = start_y,
                                bottom = end_y)
            shape = shape_pred(np_frame, box)
            if shape:
                face_chip_img = dlib.get_face_chip(np_frame, shape)
                face_descriptor = facerec.compute_face_descriptor(face_chip_img)
                name = face_recog.recognize_face(face_descriptor, threshold = 0.7)
                face_names.append(name)
    return face_names


def dalek_speak(speech):
    '''
    Break speech up into clauses and speak each one with
    various pitches, volumes and distortions
    to make the voice more Dalek like
    '''

    clauses = speech.split("|")
    for clause in clauses:
        if clause and not clause.isspace():
            if clause[:1] == ">":
                clause = clause[1:]
                pitch = PITCH_DEFAULT
                speed = SPEED_DOWN
                amplitude = AMP_UP
                sox_vol = SOX_VOL_UP
                sox_pitch = SOX_PITCH_UP
            elif clause[:1] == "<":
                clause = clause[1:]
                pitch = PITCH_DOWN
                speed = SPEED_DOWN
                amplitude = AMP_DOWN
                sox_vol = SOX_VOL_DOWN
                sox_pitch = SOX_PITCH_DOWN
            else:
                pitch = PITCH_DEFAULT
                speed = SPEED_DEFAULT
                amplitude = AMP_DEFAULT
                sox_vol = SOX_VOL_DEFAULT
                sox_pitch = SOX_PITCH_DEFAULT
            print(clause)
            cmd = ("espeak -v en-rp '%s' -p %s -s %s -a %s -z "
                   "--stdout|play -v %s - synth sine fmod 25 pitch %s" %
                   (clause, pitch, speed, amplitude, sox_vol, sox_pitch))
            os.system(cmd)


def random_msg(phrase_dict):
    '''Choose a random phrase from a list'''

    length = len(phrase_dict)
    index = random.randint(0, length-1)
    message = phrase_dict[index]
    return message


def dalek_greeting(name):
    '''Dalek will issue an appropriate greeting depending upon context'''

    greeting = ("Have a|>nice|day|>name",
                "Hello name, you are a|>friend|of the|<Darleks",
                "Greetings name",
                "Hello name",
                "name is recognized",
                "name is in the hall")
    response = random_msg(greeting)
    response = response.replace('name', name)
    print(response)
    dalek_speak(response)
    return


# Sets up a daemon thread to flash lights in line with sound
def flash_dome_lights():
    ''' Daemon thread to flash lights based on microphone noise '''

    while True:
        try:
            data = np.frombuffer(stream.read(CHUNK, False),dtype=np.int16)
            vol = abs(int(np.average(np.abs(data))))
            if vol > VOL_MIN:
                vol = vol - VOL_MIN
            else:
                vol = 0
            vol = vol * ON / VOL_MAX
            if vol > ON:
                vol =  ON
            dalek_light(DOME_LIGHTS, vol / ON)
        except ValueError:
            print ("Volume out of range: " + vol)

# start the background thread to flash the Dome Lights
p = pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK, input_device_index=2)
domeLightsThread = Thread(target=flash_dome_lights, daemon=True)
domeLightsThread.start()

dalek = Dalek()

last_message = ""
client = mqtt.Client("dalek-python")
client.connect("localhost")

# client.publish("test/message","did you get this?")
def on_message(message):
    """
    Enables the Dalek to receive a message from an Epruino Watch via
    MQTT over Bluetooth (BLE) to place it into active or inactive States
    """
    global last_message
    payload = str(message.payload.decode("utf-8"))
    if payload != last_message:
        last_message = payload
        payload = payload.replace('"', "")
        command = payload.split(",")
        print(command)
        if command[1] == "Dale" and command[2] == "face" and command[3] == "on":
            dalek.on_event('waiting')
        if command[1] == "Dale" and command[2] == "face" and command[3] == "off":
            dalek.on_event('silent')
    else:
        dalek.on_event('unknown')

client.on_message = on_message        # attach function to callback
client.subscribe("/ble/advertise/d3:fe:97:d2:d1:9e/espruino/#")

try:
    while True:
        dalek.run()
        time.sleep(0.1)
        client.loop(0.1)
except KeyboardInterrupt:
    pca.deinit()
    stream.stop_stream()
    stream.close()
    p.terminate()
    client.loop_stop()
    print("Dalek stopped by user.")

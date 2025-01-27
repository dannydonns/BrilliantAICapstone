########################################################################
# BrilliantAI Code
# Splash Screen based on spinndesign.com

########################################################################
## IMPORTS
########################################################################
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.progressbar import ProgressBar
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDToolbar
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem, TabbedPanelBase
from kivymd.uix.list import MDList, OneLineAvatarIconListItem, IRightBodyTouch
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDRoundFlatIconButton, MDFloatingActionButton
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.slider import MDSlider
from kivymd.toast import toast


#from android.permissions import request_permissions, Permission
#request_permissions([Permission.CAMERA])

import os
dirname = os.path.dirname(__file__)

import numpy as np
import cv2
import threading

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



Window.fullscreen = False

current_window = 'none'

########################################################################
## Calculation Definitions
########################################################################

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_3D(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
    

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    dist = np.linalg.norm(a - b)

    return dist

def calculate_all_features(landmarks):
    # Get coordinates
    
    foot_left = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    thumb_left = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
    foot_right = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    thumb_right = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]

    #print(landmarks)

    features = np.zeros(18)

    features[0] = calculate_angle(foot_left, ankle_left, knee_left)
    features[1] = calculate_angle(ankle_left, knee_left, hip_left)
    features[2] = calculate_angle(knee_left, hip_left, shoulder_left)
    features[3] = calculate_angle(hip_left, shoulder_left, elbow_left)
    features[4] = calculate_angle(shoulder_left, elbow_left, wrist_left)
    features[5] = calculate_angle(elbow_left, wrist_left, thumb_left)
    features[6] = calculate_angle(elbow_left, shoulder_left, shoulder_right)


    features[7] = calculate_angle(foot_right, ankle_right, knee_right)
    features[8] = calculate_angle(ankle_right, knee_right, hip_right)
    features[9] = calculate_angle(knee_right, hip_right, shoulder_right)
    features[10] = calculate_angle(hip_right, shoulder_right, elbow_right)
    features[11] = calculate_angle(shoulder_right, elbow_right, wrist_right)
    features[12] = calculate_angle(elbow_right, wrist_right, thumb_right)
    features[13] = calculate_angle(elbow_right, shoulder_right, shoulder_left)

    features[14] = calculate_distance(ankle_left, ankle_right)
    features[15] = calculate_distance(knee_left, knee_right)
    features[16] = calculate_distance(elbow_left, elbow_right)
    features[17] = calculate_distance(wrist_left, wrist_right)
    
    #os.system('clear')
    #print(features)

    return features

########################################################################
## SET WINDOW SIZE
########################################################################
Window.size = (1440, 3120)
Window.fullscreen = True

########################################################################
## LIVE CAMERA CLASS
########################################################################

class LivePoseDetect(Image):
    def __init__(self, **kwargs):
        super(LivePoseDetect, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 60)
        self.pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.allow_stretch=True
        self.counter_left = 0
        self.counter_right = 0
        self.status_left = None
        self.status_right = None

    #Drawing method to execute at intervals
    def update(self, dt):

        print(MDApp.get_running_app().root.ids.mainscreen.ids.main_screen_nav.previous_tab.name)

        if MDApp.get_running_app().root.ids.mainscreen.ids.main_screen_nav.previous_tab.name != 'live':
            return

        #Load image
        ret, frame = self.capture.read()

        # web camera
        #frame = cv2.flip(frame, 2)

        # monitor 3
        # frame = np.array(sct.grab(monitor))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Recolor image to RGB
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False

        # Make detection
        results = self.pose.process(self.image)

        # Recolor back to BGR
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        
   

        ############################ left side
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            calculate_all_features(landmarks)
            #print(angle)
            percentage = np.interp(angle, (20, 180), (100, 0))
            bar = np.interp(angle, (20, 180), (int(350/640 * frame_width), int(620/640*frame_width)))

            # Visualize angle
            cv2.putText(self.image, str(int(angle)),
                        tuple(np.multiply(elbow, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 150:
                self.status_left = "down"
            if angle < 30 and self.status_left == 'down':
                self.status_left = "up"
                self.counter_left += 1
                # print(counter)
            
            MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.left_progress_bar.value = percentage
            
            # Motion data
            cv2.putText(self.image, str(self.counter_left),
                        (int(560/640*frame_width), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 0), 1, cv2.LINE_AA)

            

            # # Status data
            # cv2.putText(image, status,
            #             (60, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(self.image, f'{int(percentage)} %',
                        (int(500/640*frame_width), int(400/480*frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), int(2*640/frame_width), cv2.LINE_AA)

            cv2.rectangle(self.image, (int(620/640*frame_width), int(420/480*frame_height)),
                          (int(350/640*frame_width), int(450/480*frame_height)), (255, 255, 255), 3)
            if angle < 30:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [0, 128, 0]
            if angle > 150:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [0, 0, 128]
            if 30 <= angle <= 150:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [128, 0, 0]

            #################### right side
            # Get coordinates
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                              # landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
                              ]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                           # landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
                           ]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                           # landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
                           ]

            

            # Calculate angle
            angle_right = calculate_angle_3D(shoulder_right, elbow_right, wrist_right)
            percentage_right = np.interp(angle_right, (20, 180), (100, 0))
            bar_right = np.interp(angle_right, (20, 180), (int(270/640 * frame_width), int(20/640*frame_width)))

            # Visualize angle
            cv2.putText(self.image, str(int(angle_right)),
                        tuple((np.multiply(elbow_right[0:2], [frame_width, frame_height])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle_right > 160:
                self.status_right = "down"
            if angle_right < 30 and self.status_right == 'down':
                self.status_right = "up"
                self.counter_right += 1
                # print(counter)

            MDApp.get_running_app().root.ids.mainscreen.ids.livescreen.ids.right_progress_bar.value = percentage_right

            # Motion data
            cv2.putText(self.image, str(self.counter_right),
                        (int(10/640*frame_width), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 0), 1, cv2.LINE_AA)

            cv2.putText(self.image, f'{int(percentage_right)} %',
                        (int(150/640*frame_width), int(400/480*frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), int(2*640/frame_width), cv2.LINE_AA)

            cv2.rectangle(self.image, (int(20 / 640 * frame_width), int(420 / 480 * frame_height)),
                          (int(270 / 640 * frame_width), int(450 / 480 * frame_height)), (255, 255, 255), 3)
            if angle_right < 30:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width):int(bar_right), :] = [0, 128, 0]
            if angle_right > 160:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width): int(bar_right), :] = [0, 0, 128]
            if 30 <= angle_right <= 160:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width): int(bar_right), :] = [128, 0, 0]

        except:
            pass

        mp_drawing.draw_landmarks(
            self.image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
        ## Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        #Convert to Kivy Texture
        buf = cv2.flip(self.image, 0).tobytes()
        texture = Texture.create(size=(self.image.shape[1], self.image.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

class LiveScreen(Screen):
    pass

########################################################################
## VIDEO CLASS
########################################################################

class VideoPoseDetect(Image):
    def __init__(self, **kwargs):
        super(VideoPoseDetect, self).__init__(**kwargs)
        #Connect to 0th camera
        #self.capture = cv2.VideoCapture('/home/vlb/Theus/APTSystemv1/assets/0918_squat_000005.mp4')
        
        self.capture = 'None'

        #Set drawing interval
        #Clock.schedule_interval(self.update, 1.0 / 60)
        self.pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.allow_stretch=True
        self.counter_left = 0
        self.counter_right = 0
        self.status_left = None
        self.status_right = None
        self.update_event = None

        self.video_length = None
        self.current_frame = 0
        self.play_flag = False
        self.play_once = False

        #Clock.schedule_once(, 1)

    def file_load(self, filename):
        try:
            self.capture = cv2.VideoCapture(filename)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video_length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
            self.current_frame = 0

            #thread = threading.Thread(target=self.process_video, args=())
            #thread.daemon = True
            #thread.start()
            self.update_event = Clock.schedule_interval(self.update, 1.0 / 60)
            self.play_flag = True
        except:
            print('File not video')
        
        # Check if video has process file
        # Load the landmarks

    def process_video(self):

        length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initial image load
        ret, frame = self.capture.read()

        frame_count = 0
        landmarks = []

        while ret:

            # web camera
            #frame = cv2.flip(frame, 2)

            # monitor 3
            # frame = np.array(sct.grab(monitor))

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            # Recolor image to RGB
            self.image = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image.flags.writeable = False

            # Make detection
            results = self.pose.process(self.image)
             
            try:
                landmarks.append(results.pose_landmarks.landmark)
                frame_count = frame_count + 1
                ret, frame = self.capture.read()
                print('Percentage: ' + str(frame_count/length * 100), flush=True)
                
            except:
                print('Cannot Process Frame, skipping ' + str(frame_count), flush=True)
                frame_count = frame_count + 1
                ret, frame = self.capture.read()
        
    def load_landmarks(self):
        print('loaded')

    #Drawing method to execute at intervals
    def update(self, dt):

        if MDApp.get_running_app().root.ids.mainscreen.ids.main_screen_nav.previous_tab.name != 'video':
            return

        if self.play_flag is False and self.play_once is False:
            return

        print(self.current_frame)
        print(self.video_length)

        if self.current_frame == self.video_length:
            return
       
        #print(app_screen_manager.ids)

        if MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.active:
            self.current_frame = int( MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.value)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame)
            
        else:
             MDApp.get_running_app().root.ids.mainscreen.ids.videoscreen.ids.video_slider.value = self.current_frame


        self.current_frame = self.current_frame+1

        #Load image
        ret, frame = self.capture.read()

        # web camera
        #frame = cv2.flip(frame, 2)

        # monitor 3
        # frame = np.array(sct.grab(monitor))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Recolor image to RGB
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False

        # Make detection
        results = self.pose.process(self.image)

        # Recolor back to BGR
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        
        

        ############################ left side
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            percentage = np.interp(angle, (20, 180), (100, 0))
            bar = np.interp(angle, (20, 180), (int(350/640 * frame_width), int(620/640*frame_width)))

            # Visualize angle
            cv2.putText(self.image, str(int(angle)),
                        tuple(np.multiply(elbow, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle > 150:
                self.status_left = "down"
            if angle < 30 and self.status_left == 'down':
                self.status_left = "up"
                self.counter_left += 1
                # print(counter)
            
            #self.lab.value = percentage
            
            # Motion data
            cv2.putText(self.image, str(self.counter_left),
                        (int(560/640*frame_width), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 0), 1, cv2.LINE_AA)

            

            # # Status data
            # cv2.putText(image, status,
            #             (60, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(self.image, f'{int(percentage)} %',
                        (int(500/640*frame_width), int(400/480*frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), int(2*640/frame_width), cv2.LINE_AA)

            cv2.rectangle(self.image, (int(620/640*frame_width), int(420/480*frame_height)),
                          (int(350/640*frame_width), int(450/480*frame_height)), (255, 255, 255), 3)
            if angle < 30:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [0, 128, 0]
            if angle > 150:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [0, 0, 128]
            if 30 <= angle <= 150:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(bar):int(620/640*frame_width), :] = [128, 0, 0]

            #################### right side
            # Get coordinates
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                              # landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
                              ]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                           # landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z
                           ]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                           # landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z
                           ]

            

            # Calculate angle
            angle_right = calculate_angle_3D(shoulder_right, elbow_right, wrist_right)
            percentage_right = np.interp(angle_right, (20, 180), (100, 0))
            bar_right = np.interp(angle_right, (20, 180), (int(270/640 * frame_width), int(20/640*frame_width)))

            # Visualize angle
            cv2.putText(self.image, str(int(angle_right)),
                        tuple((np.multiply(elbow_right[0:2], [frame_width, frame_height])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            if angle_right > 160:
                self.status_right = "down"
            if angle_right < 30 and self.status_right == 'down':
                self.status_right = "up"
                self.counter_right += 1
                # print(counter)

            #self.rab.value = percentage_right

            # Motion data
            cv2.putText(self.image, str(self.counter_right),
                        (int(10/640*frame_width), 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 0), 1, cv2.LINE_AA)

            cv2.putText(self.image, f'{int(percentage_right)} %',
                        (int(150/640*frame_width), int(400/480*frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), int(2*640/frame_width), cv2.LINE_AA)

            cv2.rectangle(self.image, (int(20 / 640 * frame_width), int(420 / 480 * frame_height)),
                          (int(270 / 640 * frame_width), int(450 / 480 * frame_height)), (255, 255, 255), 3)
            if angle_right < 30:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width):int(bar_right), :] = [0, 128, 0]
            if angle_right > 160:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width): int(bar_right), :] = [0, 0, 128]
            if 30 <= angle_right <= 160:
                self.image[int(420/480*frame_height):int(450/480*frame_height), int(20/640*frame_width): int(bar_right), :] = [128, 0, 0]

        except:
            pass

        mp_drawing.draw_landmarks(
            self.image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
        ## Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        #Convert to Kivy Texture
        buf = cv2.flip(self.image, 0).tobytes()
        texture = Texture.create(size=(self.image.shape[1], self.image.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture
        self.play_once = False

class VideoScreen(Screen):
    def build(self):
        self.manager_open = False
        #self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
        self.manager = MDFileManager(exit_manager=self.exit_manager, select_path=self.select_path)
        #self.current_dir = self.manager.current_path
        self.path = None
        #self.manager.add_widget(self.file_manager)

        self.ids.folder_button.on_release = self.file_manager_open
        self.ids.restart_button.on_release = self.restart_button_action
        self.ids.play_pause_button.on_release = self.play_button_action
        self.ids.fast_forward_button.on_release = self.forward_button_action


        self.video_update_event = None
        self.ids.video_slider.max = 0
        self.ids.video_slider.min = 0


    def file_manager_open(self):
        #if not self.manager:
            #self.manager = ModalView(size_hint=(1, 1), auto_dismiss=False)
            
            #self.manager.add_widget(self.file_manager)
        self.manager.show(self.manager.current_path)  # output manager to the screen
        self.manager_open = True
        #self.manager.open()

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''

        self.exit_manager()
        toast(path)
        self.path = path
        if self.ids.video_pose_detect.update_event is not None:
            self.ids.video_pose_detect.update_event.cancel()
            self.ids.video_pose_detect.pose = mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.ids.video_pose_detect.file_load(self.path)
        self.ids.video_slider.max = int(self.ids.video_pose_detect.capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
        

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        #self.manager.dismiss()
        self.manager.close()
        self.manager_open = False

    def play_button_action(self):
        self.ids.video_pose_detect.play_flag = not self.ids.video_pose_detect.play_flag
    def restart_button_action(self):
        self.ids.video_pose_detect.current_frame = 0
        self.ids.video_pose_detect.capture.set(cv2.CAP_PROP_POS_FRAMES,self.ids.video_pose_detect.current_frame)
        self.ids.video_pose_detect.play_once = True
    def forward_button_action(self):
        self.ids.video_pose_detect.current_frame = self.ids.video_pose_detect.current_frame + 30
        self.ids.video_pose_detect.capture.set(cv2.CAP_PROP_POS_FRAMES,self.ids.video_pose_detect.current_frame)
        self.ids.video_pose_detect.play_once = True
        

class RightCheckbox(IRightBodyTouch, MDCheckbox):
    pass

class ListItemWithCheckbox(OneLineAvatarIconListItem):
    def __init__(self, **kwargs):
        super(ListItemWithCheckbox, self).__init__(**kwargs)
        self.add_widget(RightCheckbox())

class HomeScreen(Screen):
    def build(self):
        self.pt_search = MDExpansionPanel(panel_cls=MDExpansionPanelOneLine(text='Physical Training'), icon='arm-flex')
        self.pt_list = MDList()
        self.pt_list.add_widget(ListItemWithCheckbox(text='Bicep Curl'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Jumping Jack'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Lunge'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Push-Up'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Squat'))
        self.pt_search.content=self.pt_list
        self.ids.exercise_view.add_widget(self.pt_search)

        self.pt_search = MDExpansionPanel(panel_cls=MDExpansionPanelOneLine(text='Yoga'), icon='yoga')
        self.pt_list = MDList()
        self.pt_list.add_widget(ListItemWithCheckbox(text='Bicep Curl'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Jumping Jack'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Lunge'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Push-Up'))
        self.pt_list.add_widget(ListItemWithCheckbox(text='Squat'))
        self.pt_search.content=self.pt_list
        self.ids.exercise_view.add_widget(self.pt_search)

class UserScreen(Screen):
    pass

class MainScreen(Screen):
    pass

class SplashScreen(Screen):
    pass
########################################################################
## MAIN CLASS
########################################################################
class MainApp(MDApp):
    # Global screen manager variable
    global root
    
    ########################################################################
    ## Build Function
    ########################################################################
    def build(self):
        # Set App Title
        self.title="APT App"
        # Set App Theme
        self.theme_cls.primary_palette='BlueGray'
        self.theme_cls.theme_style = "Dark"
        
        # Load kv screen files to builder
        self.root = Builder.load_file(os.path.join(dirname, 'mainScreen.kv'))
        #print(screen_manager.screen_names)

        # Other Builds
        self.root.ids.mainscreen.ids.homescreen.build()
        self.root.ids.mainscreen.ids.videoscreen.build()

        root = self.root

        
        
        # Return screen manager
        return self.root
    ########################################################################
    ## This function runs on app start
    ########################################################################
    def on_start(self):
        # Delay time for splash screen before transitioning to main screen
        Clock.schedule_once(self.change_screen, 6) # Delay for 10 seconds
        print(self.root.ids.mainscreen.ids.homescreen.ids)

    def change_window(self, win_name):
        current_window = win_name
        
    ########################################################################
    ## This function changes the current screen to main screen
    ########################################################################
    def change_screen(self, dt):    
        self.root.current = 'MainScreen'
########################################################################
## RUN APP
########################################################################      
if __name__ == '__main__':
    MainApp().run()
########################################################################
## END ==>
########################################################################



















































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
from kivy.uix.image import Image
from kivy.graphics.texture import Texture

#from android.permissions import request_permissions, Permission
#request_permissions([Permission.CAMERA])

import os
dirname = os.path.dirname(__file__)

import numpy as np
import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



Window.fullscreen = False

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
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5)

        self.allow_stretch=True

    #Drawing method to execute at intervals
    def update(self, dt):
        #Load image
        ret, self.image= self.capture.read()

        #scale_percent = 25 # percent of original size
        #width = int(self.image.shape[1] * scale_percent / 100)
        #height = int(self.image.shape[0] * scale_percent / 100)
        #dim = (width, height)
  
        # resize image
        #self.image= cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)

        #cv2.resizeself.image

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        self.image.flags.writeable = False
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(self.image)

        # Draw the pose annotation on the image.
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            self.image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        ## Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        #Convert to Kivy Texture
        buf = cv2.flip(self.image, 0).tostring()
        texture = Texture.create(size=(self.image.shape[1], self.image.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

class LiveScreen(Screen):
    def __init__(self, **kwargs):
        super(LiveScreen, self).__init__(**kwargs)

        self.layout = BoxLayout(orientation="vertical", size_hint=(None, None), size=Window.size)

        #self.field = Label(text="", size_hint=(None, None), size=(Window.width, Window.height*0.8))

        self.layout.add_widget(LivePoseDetect())
        #self.playground.add_widget(self.field)
        #self.playground.add_widget(players[1])

        self.name = 'live_screen'

        self.add_widget(self.layout)

########################################################################
## MAIN CLASS
########################################################################
class MainApp(MDApp):
    # Global screen manager variable
    global screen_manager
    screen_manager = ScreenManager()
    
    ########################################################################
    ## Build Function
    ########################################################################
    def build(self):
        # Set App Title
        self.title="APT App"
        # Set App Theme
        self.theme_cls.primary_palette='BlueGray'
        self.theme_cls.theme_style = "Dark"

        # Create Live Screen
        
        self.live_screen = LiveScreen()
        
        # Load kv screen files to builder
        screen_manager.add_widget(Builder.load_file(os.path.join(dirname, 'splashScreen.kv')))
        screen_manager.add_widget(Builder.load_file(os.path.join(dirname, 'mainScreen.kv')))
        screen_manager.add_widget(self.live_screen)
        print(screen_manager.screen_names)
        
        # Return screen manager
        return screen_manager
    ########################################################################
    ## This function runs on app start
    ########################################################################
    def on_start(self):
        # Delay time for splash screen before transitioning to main screen
        Clock.schedule_once(self.change_screen, 6) # Delay for 10 seconds
        
    ########################################################################
    ## This function changes the current screen to main screen
    ########################################################################
    def change_screen(self, dt):    
        screen_manager.current = 'live_screen'
########################################################################
## RUN APP
########################################################################      
MainApp().run()
########################################################################
## END ==>
########################################################################



















































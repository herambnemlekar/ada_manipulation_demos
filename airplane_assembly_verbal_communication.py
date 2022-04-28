#!/usr/bin/env python

import pdb
import adapy
import rospy
import sys
import time
import pickle
import numpy as np
from copy import deepcopy
from threading import Thread

from adarrt import AdaRRT
from std_msgs.msg import Float64MultiArray

import gtts
import difflib
import speech_recognition as sr
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

# from PyQt5.QtMultimedia import *
# from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


# set to False if operating real robot
sim = False

if not sim:
    # import and initialize requirements for operating real robot
    from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
    roscpp_init('airplane_assembly', [])


def createBwMatrixforTSR():
    """
    Creates the bounds matrix for the TSR.
    :returns: A 6x2 array Bw
    """
    Bw = np.zeros([6, 2])
    Bw[0, 0] = -0.005
    Bw[0, 1] = 0.005
    Bw[1, 0] = -0.005
    Bw[1, 1] = 0.005
    Bw[2, 0] = -0.005
    Bw[2, 1] = 0.005
    Bw[3, 0] = -0.005
    Bw[3, 1] = 0.005
    Bw[4, 0] = -0.005
    Bw[4, 1] = 0.005
    Bw[5, 0] = -0.005
    Bw[5, 1] = 0.005

    return Bw


def createTSR(partPose, adaHand):
    """
    Create the TSR for grasping a soda can.
    :param partPose: SE(3) transform from world to part.
    :param adaHand: ADA hand object
    :returns: A fully initialized TSR.  
    """

    # set the part TSR at the part pose
    partTSR = adapy.get_default_TSR()
    partTSR.set_T0_w(partPose)

    # transform the TSR to desired grasping pose
    partTSR_Tw_e = adaHand.get_endeffector_transform("cylinder")
    partTSR_Tw_e[2] += 0.06

    # set the transformed TSR
    partTSR.set_Tw_e(partTSR_Tw_e)
    Bw = createBwMatrixforTSR()
    partTSR.set_Bw(Bw)

    return partTSR


def toggleHand(adaHand, displacement):
    adaHand.execute_preshape(displacement)


def transition(s_from, a):
    # preconditions
    if a in [0, 1] and s_from[a] < 1:
        p = 1.0
    elif a == 2 and s_from[a] < 4 and s_from[0] == 1:
        p = 1.0
    elif a == 3 and s_from[a] < 1 and s_from[1] == 1:
        p = 1.0
    elif a == 4 and s_from[a] < 4 and s_from[a] + 1 <= s_from[a - 2]:
        p = 1.0
    elif a == 5 and s_from[a] < 1 and s_from[a] + 1 <= s_from[a - 2]:
        p = 1.0
    elif a == 6 and s_from[a] < 4:
        p = 1.0
    elif a == 7 and s_from[a] < 1 and s_from[a - 1] == 4:
        p = 1.0
    else:
        p = 0.0

    # transition to next state
    if p == 1.0:
        s_to = deepcopy(s_from)
        s_to[a] += 1
        s_to[-1] = s_from[-2]
        s_to[-2] = a
        return p, s_to
    else:
        return p, None


def speak(text):
        tts = gtts.gTTS(text)
        tts.save("robot.mp3")
        song = AudioSegment.from_mp3("robot.mp3")
        play(song)


# ------------------------------------------------------- MAIN ------------------------------------------------------- #


class Worker(QObject):
    suggested = pyqtSignal()
    commanded = pyqtSignal(str)
    finished = False

    # intialize recorder
    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        print("A moment of silence, please...")
        r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))

    def run(self):

        # # ask if the user wants the suggested part
        # speak("Do you want the suggested parts?")
        
        heard, prompt_counter = None, 0
        while (not heard) and (not self.finished) and (prompt_counter < 3):
            
            # ask if the user wants the suggested part
            speak("Do you want the suggested parts?")
            prompt_counter += 1

            with self.m as source:
                print("Speak now.")
                audio = self.r.listen(source, phrase_time_limit=5)
            print("Recognizing ...")
            value = self.r.recognize_google(audio, show_all=True)
            if value:
                for val in value["alternative"]:
                    if "yes" in val["transcript"]:
                        heard = "yes"
                        break
                    elif "no" in val["transcript"]:
                        heard = "no"
                        break

        if heard == "yes":
            self.suggested.emit()
        elif not self.finished:
            speak("Which part do you want?")
            self.listen()

    def stop(self):
        self.finished = True

    def listen(self):
        
        parts = ["long", "short", "nut", "screw", "blade", "tool", "hub", "tail wing", "main", "body"]

        part_dict = {
        "long":"long bolt", 
        "short":"short bolt", 
        "nut":"propeller nut", 
        "screw":"tail screw", 
        "blade":"propeller blades", 
        "tool":"tool", 
        "hub":"propeller hub", 
        "tail wing":"tail wing", 
        "main":"main wing", 
        "body":"airplane body"
        }

        part, ask_counter = None, 1
        while (not part) and (not self.finished) and (ask_counter < 3):
            with self.m as source:
                print("Speak now.")
                audio = self.r.listen(source, phrase_time_limit=10)
            print("Recognizing ...")
            value = self.r.recognize_google(audio, show_all=True)
            if value:
                detected = False
                for p in parts:
                    for pred in value["alternative"]:
                        print(pred["transcript"])
                        if p in pred["transcript"]:
                            detected = True
                            part = p
                if detected:
                    speak("Giving you the " + part_dict[part])
                else:
                    speak("Sorry. I didn't get that. Can you repeat?")
                    ask_counter += 1
            else:
                speak("Sorry. I didn't get that. Can you repeat?")
                ask_counter += 1

        if part and (not self.finished):
            self.commanded.emit(part)


class AssemblyController:

    def __init__(self):

        # initialize robot
        self.ada = adapy.Ada(sim)

        # ------------------------------------------ Create sim environment ---------------------------------------------- #

        # objects in airplane assembly
        storageURDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/storage.urdf"
        storagePose = [0., -0.3, -0.77, 0, 0, 0, 0]

        wingURDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/abstract_main_wing.urdf"
        wingPose = [0.75, -0.3, 0., 0.5, 0.5, 0.5, 0.5]

        tailURDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/abstract_tail_wing.urdf"
        tailPose = [-0.7, -0.25, 0.088, 0.5, 0.5, 0.5, 0.5]
        tailGraspPose = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1]]
        tailGraspOffset = [0., 0.175, 0.]

        container1URDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/container_1.urdf"
        container1_1Pose = [0.4, -0.4, 0., 0., 0., 0., 0.]
        container1_2Pose = [-0.4, -0.4, 0., 0., 0., 0., 0.]
        container1_3Pose = [0.55, -0.3, 0., 0., 0., 0., 0.]
        container1_4Pose = [-0.55, -0.3, 0., 0., 0., 0., 0.]
        container1GraspPose = [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1]]
        container1GraspOffset = [0., -0.065, -0.005]

        container2URDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/container_2.urdf"
        container2_1Pose = [0.4, -0.1, 0, 0., 0., 0., 0.]
        container2_2Pose = [-0.4, -0.1, 0., 0., 0., 0., 0.]
        container2GraspPose = [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1]]
        container2GraspOffset = [0., -0.115, 0.]
        
        container3URDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/container_3.urdf"
        container3_1Pose = [0.6, 0., 0., 0., 0., 0., 0.]
        container3_2Pose = [-0.6, 0., 0., 0., 0, 0, 0]
        container3GraspPose = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        container3GraspOffset = [0., 0., 0.]

        # hard-coded grasps
        self.graspPose, self.deliveryRotation = {}, {}
        self.graspPose["long bolts"] = [-2.11464507,  4.27069802,  2.12562682, -2.9179622, -1.1927828, -0.16230427]
        self.deliveryRotation["long bolts"] = -1.34
        self.graspPose["short bolts"] = [-0.73155659,  4.31674214,  2.28878164, -2.73375183, -1.42453116,  1.24554766]
        self.deliveryRotation["short bolts"] = 1.25
        self.graspPose["propeller nut"] = [0.49796338, 1.90442473,  3.80338018, 2.63336638,  1.44877,  1.67975607]
        self.deliveryRotation["propeller nut"] = -1.1
        self.graspPose["tail screw"] = [-0.48175263,  4.46387965,  2.68705579, -2.58115143, -1.7464862,   1.62214487]
        self.deliveryRotation["tail screw"] = 1.0  
        self.graspPose["propeller blades"] = [-2.4191907,  3.9942575,  1.29241768,  3.05926906, -0.50726387, -0.52933128]
        self.deliveryRotation["propeller blades"] = -1.1
        self.graspPose["tool"] = [-0.32843145,  4.02576609,  1.48440087, -2.87877031, -0.79457283,  1.40310179]
        self.deliveryRotation["tool"] = 1.05
        self.graspPose["propeller hub"] = [3.00773842,  4.21352853,  1.98663177, -0.17330897,  1.01156224, -0.46210507]
        self.deliveryRotation["propeller hub"] = -0.6
        self.graspPose["tail wing"] = [3.129024,  1.87404028,  3.40826295,  0.53502216, -1.86749865, -0.99044654]
        self.deliveryRotation["tail wing"] = 0.7
        

        # initialize sim environment
        self.world = self.ada.get_world()
        viewer = self.ada.start_viewer("airplane_assembly_demo", "map")

        # add parts to sim environment
        storageInWorld = self.world.add_body_from_urdf(storageURDFUri, storagePose)
        container1_1 = self.world.add_body_from_urdf(container1URDFUri, container1_1Pose)
        container1_2 = self.world.add_body_from_urdf(container1URDFUri, container1_2Pose)
        container1_3 = self.world.add_body_from_urdf(container1URDFUri, container1_3Pose)
        container1_4 = self.world.add_body_from_urdf(container1URDFUri, container1_4Pose)
        container2_1 = self.world.add_body_from_urdf(container2URDFUri, container2_1Pose)
        container2_2 = self.world.add_body_from_urdf(container2URDFUri, container2_2Pose)
        container3_1 = self.world.add_body_from_urdf(container3URDFUri, container3_1Pose)
        # container3_2 = self.world.add_body_from_urdf(container3URDFUri, container3_2Pose)
        tailWing = self.world.add_body_from_urdf(tailURDFUri, tailPose)

        # dict of all objects
        self.objects = {"long bolts": [container1_1, container1_1Pose, container1GraspPose, container1GraspOffset],
                        "short bolts": [container1_2, container1_2Pose, container1GraspPose, container1GraspOffset],
                        "propeller nut": [container1_3, container1_3Pose, container1GraspPose, container1GraspOffset],
                        "tail screw": [container1_4, container1_4Pose, container1GraspPose, container1GraspOffset],
                        "propeller blades": [container2_1, container2_1Pose, container2GraspPose, container2GraspOffset],
                        "tool": [container2_2, container2_2Pose, container2GraspPose, container2GraspOffset],
                        "propeller hub": [container3_1, container3_1Pose, container3GraspPose, container3GraspOffset],
                        "tail wing": [tailWing, tailPose, tailGraspPose, tailGraspOffset],
                        "main wing": [],
                        "airplane body": []}

        # ------------------------------------------------ Get robot config ---------------------------------------------- #

        collision = self.ada.get_self_collision_constraint()

        self.arm_skeleton = self.ada.get_arm_skeleton()
        self.arm_state_space = self.ada.get_arm_state_space()
        self.hand = self.ada.get_hand()
        self.hand_node = self.hand.get_endeffector_body_node()

        viewer.add_frame(self.hand_node)

        # ------------------------------- Start executor for real robot (not needed for sim) ----------------------------- #

        if not sim:
            self.ada.start_trajectory_executor() 

        self.armHome = [-1.57, 3.14, 1.23, -2.19, 1.8, 1.2]
        waypoints = [(0.0, self.arm_skeleton.get_positions()), (1.0, self.armHome)]
        trajectory = self.ada.compute_joint_space_path(self.arm_state_space, waypoints)
        self.ada.execute_trajectory(trajectory)
        toggleHand(self.hand, [0.15, 0.15])
      
        # -------------------------------------- Assembly and Aniticipation Info ----------------------------------------- #

        # load the learned q_values for each state
        self.qf = pickle.load(open("q_values.p", "rb"))
        self.states = pickle.load(open("states.p", "rb"))

        # actions in airplane assembly and objects required for each action
        self.remaining_user_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.action_names = ["insert main wing",
                             "insert tail wing",
                             "insert long bolts",
                             "insert tail bolt",
                             "screw long bolt",
                             "screw tail bolt",
                             "screw propeller",
                             "screw propeller base"]
        self.action_counts = [1, 1, 4, 1, 4, 1, 4, 1]
        self.required_objects = [["main wing", "airplane body"],
                                 ["tail wing", "airplane body"],
                                 ["long bolts", "tool"],
                                 ["tail screw", "tool"],
                                 ["long bolts", "tool"],
                                 ["tail screw", "tool"],
                                 ["propeller blades", "propeller hub", "short bolts", "tool"],
                                 ["propeller nut"]]
        
        # loop over all objects
        self.remaining_objects = self.objects.keys()

        # ----------------------------------------- Robot control interface ----------------------------------------- #

        # initialize GUI interface
        app = QApplication(sys.argv)
        self.initialize_app()
        app.exec_()

        # subscribe to action recognition
        sub_act = rospy.Subscriber("/april_tag_detection", Float64MultiArray, self.callback, queue_size=1)


        # simplified object names
        self.parts = {"long": "long bolts",
                      "short": "short bolts",
                      "nut": "propeller nut",
                      "screw": "tail screw",
                      "blade": "propeller blades",
                      "tool": "tool",
                      "hub": "propeller hub",
                      "tail wing": "tail wing",
                      "main": "main wing",
                      "body": "airplane body"}

        # initialize user sequence
        self.time_step = 0
        self.user_sequence = []

        self.time_to_respond = 10


    def initialize_app(self):

        self.win = QMainWindow()
        self.win.setWindowTitle("Robot Commander")
        self.win.setGeometry(0, 0, 1280, 720)

        instruct_button = QPushButton(self.win)
        instruct_button.setText("Instructions")
        instruct_button.setFont(QFont('Arial', 24))
        instruct_button.setGeometry(600, 400, 600, 60)
        instruct_button.clicked.connect(self.show_instructions)

        start_button = QPushButton(self.win)
        start_button.setText("Start Assembly")
        start_button.setFont(QFont('Arial', 24))
        start_button.setGeometry(600, 500, 600, 60)
        start_button.clicked.connect(self.win.close)

        self.win.showMaximized()
    

    def show_instructions(self):
        # QMediaPlayer
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile('runyu_fulltask.mp4')))

        # Set widget
        self.videoWidget = QVideoWidget()
        # self.videoWidget.setGeometry(self.pos().x(), self.pos().y(), self.width(), self.height())
        self.setCentralWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # Play
        self.mediaPlayer.play()
        
    
    def callback(self, data):

        # current recognised action sequence
        detected_sequence = [int(a) for a in data.data]

        # current recognised parts
        detected_parts = data.layout.dim[0].label.split(",")

        # wait for a new action to be detected
        # if len(detected_sequence) > len(self.user_sequence) or self.time_step==0:
            
        # update action sequence
        self.user_sequence = detected_sequence
        self.time_step = len(self.user_sequence)

        # determine current state based on detected action sequence
        current_state = self.states[0]
        for user_action in self.user_sequence:
            for i in range(self.action_counts[user_action]):
                p, next_state = transition(current_state, user_action)
                current_state = next_state      


        # update remaining parts
        self.remaining_objects = [rem_obj for rem_obj in self.remaining_objects if rem_obj not in detected_parts]        


        # ---------------------------------------- Anticipate next user action --------------------------------------- #
        sensitivity = 0.0
        max_action_val = -np.inf
        available_actions, anticipated_actions = [], []
        
        for a in self.remaining_user_actions:
            s_idx = self.states.index(current_state)
            p, next_state = transition(current_state, a)
            
            # check if the action results in a new state
            if next_state:
                available_actions.append(a)

                if self.qf[s_idx][a] > (1 + sensitivity) * max_action_val:
                    anticipated_actions = [a]
                    max_action_val = self.qf[s_idx][a]

                elif (1 - sensitivity) * max_action_val <= self.qf[s_idx][a] <= (1 + sensitivity) * max_action_val:
                    anticipated_actions.append(a)
                    max_action_val = self.qf[s_idx][a]

        # determine the legible names of the anticipated actions
        self.anticipated_action_names = [self.action_names[a] for a in anticipated_actions]

        # determine objects required for anticipated actions
        suggested_objects = []
        for a in anticipated_actions:
            suggested_objects += [obj for obj in self.required_objects[a] if obj in self.remaining_objects]
        self.suggested_objects = list(set(suggested_objects))

        # ----------------------------------------- Robot control interface ----------------------------------------- #

        # initialize GUI interface
        app = QApplication(sys.argv)
        self.start_assembly()
        # QTimer.singleShot(12000, self.win.close)
        app.exec_()


    def start_assembly(self):

        options = deepcopy(self.remaining_objects)
        suggestions = deepcopy(self.suggested_objects)
        suggestion_text = deepcopy(self.anticipated_action_names)

        self.win = QMainWindow()
        
        # window title and size
        self.win.setWindowTitle("Robot Commander")
        self.win.setGeometry(0, 0, 1280, 720)

        # prompt
        self.win.query = QLabel(self.win)
        self.win.query.setText("Select the parts you want the robot to deliver:")
        self.win.query.setFont(QFont('Arial', 24))
        self.win.query.adjustSize()
        self.win.query.move(75, 75)

        # task info
        self.win.image = QLabel(self.win)
        pixmap = QPixmap('task.jpg')
        pixmap = pixmap.scaledToWidth(950)
        self.win.image.setPixmap(pixmap)
        self.win.image.adjustSize()
        self.win.image.move(825, 200)

        # inputs
        self.win.options = []
        self.win.user_choice = []
        self.win.act = False

        # print current time step
        self.win.step = QLabel(self.win)
        self.win.step.setText("Current time step: " + str(self.time_step))
        self.win.step.setFont(QFont('Arial', 36))
        self.win.step.adjustSize()
        self.win.step.move(820, 125)

        # pre-text for suggestion action
        self.win.pre_text = QLabel(self.win)
        self.win.pre_text.setText("Suggested next action:")
        self.win.pre_text.setFont(QFont('Arial', 36))
        self.win.pre_text.adjustSize()
        self.win.pre_text.move(820, 750)

        # print the anticipated action
        self.win.user_instruction = QLabel(self.win)
        self.win.user_instruction.setText(str(suggestion_text))
        self.win.user_instruction.setFont(QFont('Arial', 32))
        self.win.user_instruction.adjustSize()
        self.win.user_instruction.move(1340, 750)
        self.win.user_instruction.setStyleSheet("color: green")


        # add option of performing no action
        options += ["none"]

        # print the options
        option_x, option_y = 260, 150
        buttons = []
        for opt in options:
            buttons.append(QPushButton(self.win))
            buttons[-1].setText(opt)
            buttons[-1].setFont(QFont('Arial', 20))
            buttons[-1].setGeometry(option_x, option_y, 225, 50)
            buttons[-1].setCheckable(True)
            # buttons[-1].clicked.connect(self.win.set_choice)
            if opt in suggestions:
                buttons[-1].setStyleSheet("QPushButton {background-color : lightgreen;} QPushButton::checked {background-color : lightpink;}")
            else:
                buttons[-1].setStyleSheet("QPushButton::checked {background-color : lightpink;}")
            option_y += 50    
        self.win.options = buttons

        # button for performing suggested actions
        option_x = 130
        option_y += 50
        self.win.suggested_button = QPushButton(self.win)
        self.win.suggested_button.setText("YES. Give the parts you suggested.")
        self.win.suggested_button.setFont(QFont('Arial', 20))
        self.win.suggested_button.setGeometry(option_x, option_y, 500, 50)
        self.win.suggested_button.setStyleSheet("background-color : lightgreen")
        self.win.suggested_button.setCheckable(True)
        self.win.suggested_button.clicked.connect(self.deliver_suggested)

        # button for performing selected actions
        option_x = 130
        option_y += 75
        self.win.selected_button = QPushButton(self.win)
        self.win.selected_button.setText("NO. Give the parts I selected.")
        self.win.selected_button.setFont(QFont('Arial', 20))
        self.win.selected_button.setGeometry(option_x, option_y, 500, 50)
        self.win.selected_button.setStyleSheet("background-color : lightpink")
        self.win.selected_button.setCheckable(True)
        self.win.selected_button.clicked.connect(self.deliver_selected)

        # start timer
        self.set_timer()

        # show full-screen window
        self.win.showMaximized()

        # thread for verbal communication
        self.worker = Worker()
        self.thread = QThread(self.win)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.suggested.connect(self.deliver_suggested)
        self.worker.suggested.connect(self.worker.deleteLater)
        self.worker.commanded.connect(self.deliver_commanded)
        self.worker.commanded.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()


    def set_timer(self):
        self.time_left = deepcopy(self.time_to_respond)
        self.win.countdown = QLabel(self.win)
        self.win.countdown.setText(str(self.time_left))
        self.win.countdown.setFont(QFont('Arial', 36))
        self.win.countdown.setStyleSheet("background-color: khaki")
        self.win.countdown.adjustSize()
        self.win.countdown.move(1720, 125)
        self.win.timer = QTimer()
        self.win.timer.timeout.connect(self.timer_update)
        self.win.timer.start(1000)


    def timer_update(self): 
        self.time_left -=1
        self.win.countdown.setText(" " + str(self.time_left) + " ")
        if self.time_left == 0:
            self.time_left = deepcopy(self.time_to_respond)
            self.win.countdown.setText(str(self.time_left))


    def deliver_suggested(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.thread.terminate()
        self.deliver_part(self.suggested_objects)

    def deliver_selected(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.thread.terminate()
        selected_parts = []
        for option in self.win.options:
                if option.isChecked():
                    selected_parts.append(option.text())
        self.deliver_part(selected_parts)

    def deliver_commanded(self, part):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.thread.terminate()
        commanded_part = self.parts[part]
        self.deliver_part([commanded_part])

    def deliver_part(self, objects_to_deliver):

        # loop over all objects to be delivered
        for chosen_obj in objects_to_deliver:

            # instruct the user to retreive the parts that cannot be delivered by the robot
            if chosen_obj in ["main wing", "airplane body", "none"]:
                print("Cannot provide this part.")
                msg = QMessageBox()
                msg.setText("Get the parts you need while the robot waits.")
                msg.setFont(QFont('Arial', 20))
                msg.setWindowTitle("Robot Message")
                QTimer.singleShot(10000, msg.close)    
                msg.exec_()

            # deliver parts requested by the user whenever possible
            else:
                print("Providing the required part.")

                # grasp TSR for object
                obj = self.objects[chosen_obj][0]
                objPose = self.objects[chosen_obj][1]
                objGraspPose = self.objects[chosen_obj][2]
                objOffset = self.objects[chosen_obj][3]
                objPoseMat = [objGraspPose[0] + [objPose[0] + objOffset[0]],
                              objGraspPose[1] + [objPose[1] + objOffset[1]],
                              objGraspPose[2] + [objPose[2] + objOffset[2]],
                             [0.0, 0.0, 0.0, 1.0]]
                objTSR = createTSR(objPoseMat, self.hand)
                # marker = viewer.add_tsr_marker(objTSR)
                # raw_input("Does the marker look good?")

                # ---------------------------------------- Collision detection --------------------------------------- #

                # collision_free_constraint = self.ada.set_up_collision_detection(ada.get_arm_state_space(), self.ada.get_arm_skeleton(),
                #                                                                        [obj])
                # full_collision_constraint = self.ada.get_full_collision_constraint(ada.get_arm_state_space(),
                #                                                                      self.ada.get_arm_skeleton(),
                #                                                                      collision_free_constraint)
                # collision = self.ada.get_self_collision_constraint()


                # -------------------------------------- Plan path for grasping -------------------------------------- #
                
                # use pre-computed grasp configuration if available
                if chosen_obj in self.graspPose.keys():
                    print("Running hard-coded.")
                    grasp_configuration = self.graspPose[chosen_obj]
                else:
                    # perform IK to compute grasp configuration
                    ik_sampleable = adapy.create_ik(self.arm_skeleton, self.arm_state_space, objTSR, self.hand_node)
                    ik_generator = ik_sampleable.create_sample_generator()
                    configurations = []
                    samples = 0
                    maxSamples = 1
                    print("Finding IK configuration...")
                    while samples < maxSamples and ik_generator.can_sample():
                        goal_state = ik_generator.sample(self.arm_state_space)
                        if len(goal_state) == 0:
                            continue
                        else:
                            samples += 1
                        configurations.append(goal_state)   
                    
                    print("Found new configuration.")
                    grasp_configuration = configurations[0]

                # plan path to grasp configuration
                waypoints = [(0.0, self.armHome),(1.0, grasp_configuration)]
                trajectory = self.ada.compute_joint_space_path(self.ada.get_arm_state_space(), waypoints)

                # ------------------------------------------ Execute path to grasp object --------------------------------- #

                if not trajectory:
                    print("Failed to find a solution!")
                else:
                    # execute the planned trajectory
                    self.ada.execute_trajectory(trajectory)
                    
                    # lower the griper to grasp object
                    waypoints = []
                    for i in range(3):
                        jac = self.arm_skeleton.get_linear_jacobian(self.hand_node)
                        full_jac = self.arm_skeleton.get_jacobian(self.hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, 0.025])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = self.arm_skeleton.get_positions()
                        new_q = q + delta_q
                        waypoints.append((i, new_q))
                        self.ada.set_positions(new_q)

                    traj = self.ada.compute_joint_space_path(self.ada.get_arm_state_space(), waypoints)
                    self.ada.execute_trajectory(traj)
                    
                    # grasp the object                    
                    toggleHand(self.hand, [1.5, 1.5])
                    time.sleep(1.5)
                    self.hand.grab(obj)

                    # ----------------------- Lift up grasped object using Jacobian pseudo-inverse ------------------------ #

                    waypoints = []
                    for i in range(6):
                        jac = self.arm_skeleton.get_linear_jacobian(self.hand_node)
                        full_jac = self.arm_skeleton.get_jacobian(self.hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, -0.025])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = self.arm_skeleton.get_positions()
                        new_q = q + delta_q
                        waypoints.append((i, new_q))
                        self.ada.set_positions(new_q)

                    print("Picking up.")
                    traj = self.ada.compute_joint_space_path(self.ada.get_arm_state_space(), waypoints)
                    self.ada.execute_trajectory(traj)

                    # ----------------------------------- Move grasped object to workbench ------------------------------- #
                    
                    current_position = self.arm_skeleton.get_positions()
                    new_position = current_position.copy()
                    new_position[0] += self.deliveryRotation[chosen_obj]
                    waypoints = [(0.0, current_position), (1.0, new_position)]
                    traj = self.ada.compute_joint_space_path(self.ada.get_arm_state_space(), waypoints)
                    self.ada.execute_trajectory(traj)

                    # ----------------------- Lower grasped object using Jacobian pseudo-inverse ------------------------ #

                    waypoints = []
                    for i in range(4):
                        jac = self.arm_skeleton.get_linear_jacobian(self.hand_node)
                        full_jac = self.arm_skeleton.get_jacobian(self.hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, 0.025])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = self.arm_skeleton.get_positions()
                        new_q = q + delta_q
                        waypoints.append((i, new_q))
                        self.ada.set_positions(new_q)

                    print("Keeping down.")
                    traj = self.ada.compute_joint_space_path(self.ada.get_arm_state_space(), waypoints)
                    self.ada.execute_trajectory(traj)

                    self.hand.ungrab(obj)
                    toggleHand(self.hand, [0.15, 0.15])
                    self.world.remove_skeleton(obj)
                    time.sleep(1)

                    # ------------------- Move robot back to home ------------------- #

                    waypoints = [(0.0, self.arm_skeleton.get_positions()), (1.0, self.armHome)]
                    traj = self.ada.compute_joint_space_path(self.ada.get_arm_state_space(), waypoints)
                    self.ada.execute_trajectory(traj)

        print("Finished executing actions.")
        
        # self.remaining_objects = [rem_obj for rem_obj in self.remaining_objects if rem_obj not in objects_to_deliver]

        time.sleep(3)
        self.win.close()



# MAIN
# initialise ros node
rospy.init_node("adapy_assembly")
ac = AssemblyController()
try:
    rospy.spin()
except KeyboardInterrupt:
    print ("Shutting down")

raw_input("Press Enter to Quit...")
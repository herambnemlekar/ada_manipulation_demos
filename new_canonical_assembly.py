#!/usr/bin/env python3

import pdb
import sys
import time
import pickle
import numpy as np
from copy import deepcopy
from threading import Thread

import adapy
import rospy
from std_msgs.msg import Float64MultiArray
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import common


# set to False if operating real robot
IS_SIM = False

# directory path for each machine
directory_syspath = "/home/icaros/ros_ws/src/ada_manipulation_demos"

# urdf files path 
urdf_filepath = "package://ada_manipulation_demos/urdf_collection"


# ------------------------------------------------------- MAIN ------------------------------------------------------- #

class AssemblyController(QMainWindow):

    def __init__(self):
        super(AssemblyController, self).__init__()

        # initialize robot
        self.ada = adapy.Ada(IS_SIM)

        # ------------------------------------------ Create sim environment ---------------------------------------------- #

        # objects in airplane assembly
        storageURDFUri = urdf_filepath + "/storage.urdf"
        storagePose = [0., -0.3, -0.77, 0, 0, 0, 0]

        wingURDFUri = urdf_filepath + "/abstract_main_wing.urdf"
        wingPose = [0.75, -0.3, 0., 0.5, 0.5, 0.5, 0.5]

        tailURDFUri = urdf_filepath + "/abstract_tail_wing.urdf"
        tailPose = [-0.7, -0.25, 0.088, 0.5, 0.5, 0.5, 0.5]

        container1URDFUri = urdf_filepath + "/container_1.urdf"
        container1_1Pose = [0.4, -0.4, 0., 0., 0., 0., 0.]
        container1_2Pose = [-0.4, -0.4, 0., 0., 0., 0., 0.]
        container1_3Pose = [0.55, -0.3, 0., 0., 0., 0., 0.]
        container1_4Pose = [-0.55, -0.3, 0., 0., 0., 0., 0.]

        container2URDFUri = urdf_filepath + "/container_2.urdf"
        container2_1Pose = [0.4, -0.1, 0, 0., 0., 0., 0.]
        container2_2Pose = [-0.4, -0.1, 0., 0., 0., 0., 0.]
        
        container3URDFUri = urdf_filepath + "/container_3.urdf"
        container3_1Pose = [0.6, 0., 0., 0., 0., 0., 0.]
        container3_2Pose = [-0.6, 0., 0., 0., 0, 0, 0]

        # grasp TSR and offsets
        tailGraspPose = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1]]
        tailGraspOffset = [0., 0.175, 0.]
        container1GraspPose = [[0., 1., 0., 0.], [1., 0., 0., -0.05], [0., 0., -1, 0.8], [0., 0., 0., 1.]]
        container1GraspOffset = [0., 0., -0.07]
        container2GraspPose = [[0., 1., 0., 0.], [1., 0., 0., -0.1], [0., 0., -1, 0.1], [0., 0., 0., 1.]]
        container2GraspOffset = [0., -0.115, 0.]
        container3GraspPose = [[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.1], [0., 0., 0., 1.]]
        container3GraspOffset = [0., 0., 0.]

        # hard-coded grasps
        self.graspConfig, self.deliveryRotation, self.deliveryHandRotation, self.lowerDistance, self.liftOffset = {}, {}, {}, {}, {}

        self.graspConfig["long bolts"] = [-2.06624655,  4.37198852,  2.3886246,  -2.84061763,  4.90123373, -6.59571791]
        self.graspConfig["long bolts back"] = [-2.06807695,  4.2959132,   2.41701177, -2.789237,   11.03463327, -6.56036019]
        #[-2.09854350, 4.27745083, 2.12565941, -2.91795263, -1.19278937, -0.39783092]
        self.deliveryRotation["long bolts"] = -1.85
        self.deliveryHandRotation["long bolts"] = -1.65
        self.lowerDistance["long bolts"] = -0.14
        self.liftOffset["long bolts"] = [0., 0., 0.165]
        # self.graspConfig["long bolts"] = [-2.11464507,  4.27069802,  2.12562682, -2.9179622, -1.1927828, -0.16230427]
        # self.deliveryRotation["long bolts"] = -1.34

        # self.graspConfig["short bolts"] = [-0.72561783, 4.31588712, 2.28856202, -2.71514972, -1.42200445, 1.01089267]
        # self.deliveryRotation["short bolts"] = 1.25
        self.graspConfig["short bolts"] = [ -7.07682948,   4.45124074,   2.65111774,  -2.60687012,  17.12044212, -11.50343272]
        #[-0.72561783, 4.31588712, 2.28856202, -2.71514972, -1.42200445, 1.01089267]
        self.graspConfig["short bolts back"] = [ -7.07814016,   4.39303122,   2.67651871,  -2.55499361,  16.98924735,  -11.45468195]
        self.deliveryRotation["short bolts"] = 1.85
        self.deliveryHandRotation["short bolts"] = 1.35
        self.lowerDistance["short bolts"] = -0.17
        self.liftOffset["short bolts"] = [0., 0., 0.2]


        # self.graspConfig["short wire"] = [0.49700125, 1.86043184, 3.78425230, 2.63384048, 1.44808279, 1.67817618]
        # self.deliveryRotation["short wire"] = -1.1
        # using propeller nut location for short wire
        self.graspConfig["short wire"] = [-2.03877631, 4.09967790, 1.60438025, -0.19636232, 0.71718155, 2.21799853]
        self.graspConfig["short wire lower"] = [-2.03990137,  4.21760047,  1.61261608, -0.17346979,  0.6146171,   2.24402321]
        self.deliveryRotation["short wire"] = -2.5
        self.deliveryHandRotation["short wire"] = -2.25
        self.lowerDistance["short wire"] = -0.14
        self.liftOffset["short wire"] = [0., 0., 0.165]

        self.graspConfig["long wire"] = [-0.46015322, 4.47079882, 2.68192519, -2.584758426, -1.74260217, 1.457295330]
        #[-0.46015322, 4.47079882, 2.68192519, -2.584758426, -1.74260217, 1.457295330]
        self.deliveryRotation["long wire"] = 1.6
        self.deliveryHandRotation["long wire"] = 1.35
        self.lowerDistance["long wire"] = -0.18
        self.liftOffset["long wire"] = [0., 0., 0.2]
        #1.0  

        # self.graspConfig["small box"] = [-2.4191907,  3.9942575,  1.29241768,  3.05926906, -0.50726387, -0.52933128]
        self.graspConfig["small box"] =[-2.44054132, 3.90477552, 1.44756147, -2.95127490, -0.86210359, -0.75472121]
        self.deliveryRotation["small box"] = -1.4
        self.deliveryHandRotation["small box"] = -1.6
        self.liftOffset["small box"] = [0.14, 0., 0.165]

        # self.graspConfig["large box"] = [3.129024,  1.87404028,  3.40826295,  0.53502216, -1.86749865, -0.99044654]
        self.graspConfig["large box"] = [3.11715628, 1.88217232, 3.35777661, 0.68723824, -1.99417529, -0.89891913]
        self.deliveryRotation["large box"] = 1.0
        self.liftOffset["large box"] = [0., 0., 0.1]

        # self.graspConfig["tool"] = [-0.32843145,  4.02576609,  1.48440087, -2.87877031, -0.79457283,  1.40310179]
        # self.deliveryRotation["tool"] = 1.05
        self.graspConfig["tool"] = [-0.39286100, 4.03060775, 1.54378641, -2.93910479, -0.79306859, 1.08912509]
        self.deliveryRotation["tool"] = 1.05 
        self.lowerDistance["tool"] = -0.05
        self.liftOffset["tool"] = [0., 0., 0.165]

        self.graspConfig["propeller hub"] = [3.00773842,  4.21352853,  1.98663177, -0.17330897,  1.01156224, -0.46210507]
        self.deliveryRotation["propeller hub"] = -0.6

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
                        "short wire": [container1_3, container1_3Pose, container1GraspPose, container1GraspOffset],
                        "long wire": [container1_4, container1_4Pose, container1GraspPose, container1GraspOffset],
                        "small box": [container2_1, container2_1Pose, container2GraspPose, container2GraspOffset],
                        "tool": [container2_2, container2_2Pose, container2GraspPose, container2GraspOffset],
                        # "propeller hub": [container3_1, container3_1Pose, container3GraspPose, container3GraspOffset],
                        "large box": [tailWing, tailPose, tailGraspPose, tailGraspOffset]}

        # ------------------------------------------------ Get robot config ---------------------------------------------- #

        collision = self.ada.get_self_collision_constraint()

        self.arm_skeleton = self.ada.get_arm_skeleton()
        self.arm_state_space = self.ada.get_arm_state_space()
        self.hand = self.ada.get_hand()
        self.hand_node = self.hand.get_end_effector_body_node()

        viewer.add_frame(self.hand_node)

        # ------------------------------- Start executor for real robot (not needed for sim) ----------------------------- #

        if not IS_SIM:
            self.ada.start_trajectory_controllers() 

        self.armHome = [-1.57, 3.14, 1.23, -2.19, 1.8, 1.2]
        waypoints = [(0.0, self.arm_skeleton.get_positions()), (1.0, self.armHome)]
        trajectory = self.ada.compute_joint_space_path(waypoints)  # self.ada.plan_to_configuration(self.armHome)
        self.ada.execute_trajectory(trajectory)
        self.hand.execute_preshape([0.15, 0.15])
      
        # ------------------------------------------------- Assembly Info ------------------------------------------------ #
        
        # objects yet to be delivered
        self.remaining_objects = list(self.objects.keys())

        # subscribe to action recognition
        sub_act = rospy.Subscriber("/april_tag_detection", Float64MultiArray, self.callback, queue_size=1)

        # initialize user sequence
        self.time_step = 0
        self.user_sequence = []
        self.anticipated_action_name = []
        self.suggested_objects = []

        # ------------------------------------------------ GUI details --------------------------------------------------- #

        # window title and size
        self.setWindowTitle("Robot Commander")
        self.setGeometry(0, 0, 1280, 720)

        # prompt
        query = QLabel(self)
        query.setText("Which part(s) do you want?")
        query.setFont(QFont('Arial', 28))
        query.adjustSize()
        query.move(95, 135)

        # task info
        assembly_image = QLabel(self)
        pixmap = QPixmap(directory_syspath + "/src/canonical_task.png")
        pixmap = pixmap.scaledToWidth(1125)
        assembly_image.setPixmap(pixmap)
        assembly_image.adjustSize()
        assembly_image.move(660, 145)

        # inputs
        options = deepcopy(self.remaining_objects)

        # print the options
        option_x, option_y = 210, 200
        buttons = []
        for opt in options:
            opt_button = QPushButton(self)
            opt_button.setText(opt)
            opt_button.setFont(QFont('Arial', 20))
            opt_button.setGeometry(option_x, option_y, 225, 50)
            opt_button.setCheckable(True)
            opt_button.setStyleSheet("QPushButton::checked {background-color : lightpink;}")
            buttons.append(opt_button)
            option_y += 50    
        self.option_buttons = buttons

        # button for performing selected actions
        option_x = 85
        option_y += 60
        self.selected_button = QPushButton(self)
        self.selected_button.setText("Give me the selected parts.")
        self.selected_button.setFont(QFont('Arial', 20))
        self.selected_button.setGeometry(option_x, option_y, 500, 50)
        self.selected_button.setStyleSheet("background-color : lightpink")
        self.selected_button.setCheckable(True)
        self.selected_button.clicked.connect(self.deliver_part)

        # print current time step
        self.step_label = QLabel(self)
        self.step_label.setText("Current time step: " + str(self.time_step))
        self.step_label.setFont(QFont('Arial', 36))
        self.step_label.adjustSize()
        self.step_label.move(715, 65)

        # update timer
        self.time_to_respond = 10
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_application)
        
        self.time_left = deepcopy(self.time_to_respond)
        self.countdown = QLabel(self)
        self.countdown.setText(str(self.time_left))
        self.countdown.setFont(QFont('Arial', 36))
        self.countdown.setStyleSheet("background-color: khaki")
        self.countdown.adjustSize()
        self.countdown.move(1720, 65)
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.timer_update)
        
        self.timer.start(self.time_to_respond*1000) 
        self.countdown_timer.start(1000)


    def timer_update(self): 
        self.time_left -=1
        self.countdown.setText(" " + str(self.time_left) + " ")
        if self.time_left == 0:
            self.time_left = deepcopy(self.time_to_respond)
            self.countdown.setText(str(self.time_left))


    def update_application(self):

        # update time stamp
        self.step_label.setText("Current time step: " + str(self.time_step))

        # update suggested options
        for opt_button in self.option_buttons:
            opt_button.setChecked(False)
            if opt_button.text() not in self.remaining_objects:
                opt_button.setChecked(False)
                opt_button.setCheckable(False)
                opt_button.setStyleSheet("QPushButton {color : lightgrey;}")
            else:
                opt_button.setStyleSheet("QPushButton::checked {background-color : lightpink;}")

        # update action buttons
        self.selected_button.setChecked(False)
        
        
    def callback(self, data):

        # current recognised action sequence
        detected_sequence = [int(a) for a in data.data]

        # current recognised parts
        detected_parts = data.layout.dim[0].label.split(",")
            
        # update action sequence
        self.user_sequence = detected_sequence
        self.time_step = len(self.user_sequence)

        # update remaining parts
        self.remaining_objects = [rem_obj for rem_obj in self.remaining_objects if rem_obj not in detected_parts]        


    def deliver_part(self):
        
        # check which objects were selected by the user       
        if self.selected_button.isChecked():
            objects_to_deliver = []
            for option in self.option_buttons:
                if option.isChecked():
                    objects_to_deliver.append(option.text())
        else:
            objects_to_deliver = []
        
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
            else:
                # deliver parts requested by the user whenever possible
                print("Providing the required part.")

                # ---------------------------------------- Collision detection --------------------------------------- #

                # collision_free_constraint = self.ada.set_up_collision_detection(ada.get_arm_state_space(), self.ada.get_arm_skeleton(),
                #                                                                        [obj])
                # full_collision_constraint = self.ada.get_full_collision_constraint(ada.get_arm_state_space(),
                #                                                                      self.ada.get_arm_skeleton(),
                #                                                                      collision_free_constraint)
                # collision = self.ada.get_self_collision_constraint()


                # -------------------------------------- Plan path for grasping -------------------------------------- #
                
                obj = self.objects[chosen_obj][0]
                objPose = self.objects[chosen_obj][1]
                objGraspPose = self.objects[chosen_obj][2]

                # use pre-computed grasp configuration if available
                if chosen_obj in self.graspConfig.keys():
                    print("Running hard-coded...")
                    grasp_configuration = self.graspConfig[chosen_obj]
                else:
                    print("Creating new TSR.")
                    # grasp TSR for object
                    objTSR = common.createTSR(objPose, objGraspPose)
                    # marker = viewer.add_tsr_marker(objTSR)
                    # input("Marker look good?")

                    # perform IK to compute grasp configuration
                    ik_sampleable = adapy.create_ik(self.arm_skeleton, self.arm_state_space, objTSR, self.hand_node)
                    ik_generator = ik_sampleable.create_sample_generator()
                    configurations = []
                    samples, max_samples = 0, 10
                    while samples < max_samples and ik_generator.can_sample():
                        samples += 1
                        goal_state = ik_generator.sample(self.arm_state_space)
                        if len(goal_state) == 0:
                            continue
                        configurations.append(goal_state)
                        print("Found new configuration.")

                    grasp_configuration = configurations[0]

                # plan path to grasp configuration
                waypoints = [(0.0, self.armHome),(1.0, grasp_configuration)]
                trajectory = self.ada.compute_joint_space_path(waypoints)


                # ------------------------------------------ Execute path to grasp object --------------------------------- #

                if not trajectory:
                    print("Failed to find a solution!")
                else:
                    # execute the planned trajectory
                    self.ada.execute_trajectory(trajectory)



                # test
                # traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0.1, 0.1, 0.12])
                # self.ada.execute_trajectory(traj)
                # current_position = self.arm_skeleton.get_positions()
                # print(current_position)

                # input("Press ENTER to continue ...")

                    
                # lower gripper
                if chosen_obj == "long wire":
                    traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., -0.04])
                else:
                    traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., -0.045])
                self.ada.execute_trajectory(traj)
                
                # grasp the object                    
                self.hand.execute_preshape([1.3, 1.3])
                time.sleep(1.5)
                self.hand.grab(obj)

                # lift up grasped object
                # if chosen_obj in ["short bolts", "long wire"]:
                #     traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., 0.2])
                # elif chosen_obj =="large box":
                #     traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., 0.1])
                # elif chosen_obj == "small box":
                #     traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0.12, 0., 0.165])
                # else:
                #     
                traj = self.ada.plan_to_offset("j2n6s200_hand_base", self.liftOffset[chosen_obj])
                self.ada.execute_trajectory(traj)

                # move grasped object to workbench
                if chosen_obj in ["tool"]:
                    # move forward
                    traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0.55, 0.])
                    self.ada.execute_trajectory(traj)
                elif chosen_obj in ["long wire", "short wire", "long bolts", "short bolts", "small box"]:
                    current_position = self.arm_skeleton.get_positions()
                    new_position = current_position.copy()
                    new_position[0] += self.deliveryRotation[chosen_obj]
                    new_position[5] += self.deliveryHandRotation[chosen_obj]
                    waypoints = [(0.0, current_position), (1.0, new_position)]
                    traj = self.ada.compute_joint_space_path(waypoints)
                    self.ada.execute_trajectory(traj)

                else:
                    current_position = self.arm_skeleton.get_positions()
                    new_position = current_position.copy()
                    new_position[0] += self.deliveryRotation[chosen_obj]
                    waypoints = [(0.0, current_position), (1.0, new_position)]
                    traj = self.ada.compute_joint_space_path(waypoints)
                    self.ada.execute_trajectory(traj)

                # ----------------------- Lower grasped object using Jacobian pseudo-inverse ------------------------ #

                # traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., -0.125])
                # self.ada.execute_trajectory(traj)

                # ------------------- Hold Container for human to grab ------------------- # 
                # test
                # time.sleep(3)

                # ungrab if small box and large box
                if chosen_obj in ["small box","large box"]:
                    if chosen_obj == "small box":
                        traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., -0.12])
                    else:
                        traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., -0.08])
                    self.ada.execute_trajectory(traj)
                    # ungrab
                    self.hand.ungrab()
                    self.hand.execute_preshape([0.15, 0.15])
                    self.world.remove_skeleton(obj)
                    time.sleep(1)

                else:
                # ------------------- Move Container back to original place if not boxes------------------- #
                    # wait for user to grab item
                    time.sleep(3)

                    if chosen_obj in ["long bolts", "short bolts", "long wire"]:
                        #print("smoothier turn")
                        current_position = self.arm_skeleton.get_positions()
                        new_position = current_position.copy()
                        new_position[0] -= self.deliveryRotation[chosen_obj]
                        new_position[5] -= self.deliveryHandRotation[chosen_obj]
                        waypoints = [(0.0, current_position),(1.0, new_position)]
                        trajectory = self.ada.compute_joint_space_path(waypoints)
                        # if not trajectory:
                        #     print("Failed to find a solution!")
                        # else:
                        #     # execute the planned trajectory
                        #     self.ada.execute_trajectory(trajectory)
                        # current_position = self.arm_skeleton.get_positions()
                        # waypoints = [(0.0, current_position),(3.0, grasp_configuration)]
                        # trajectory = self.ada.compute_joint_space_path(waypoints)
                    elif chosen_obj in ["short wire"]:
                        current_position = self.arm_skeleton.get_positions()
                        new_position = current_position.copy()
                        new_position[0] -= self.deliveryRotation[chosen_obj]
                        new_position[5] -= self.deliveryHandRotation[chosen_obj]
                        waypoints = [(0.0, current_position),(1.0, new_position)]
                        trajectory = self.ada.compute_joint_space_path(waypoints)
                    else:
                        current_position = self.arm_skeleton.get_positions()
                        waypoints = [(0.0, current_position),(1.0, grasp_configuration)]
                        trajectory = self.ada.compute_joint_space_path(waypoints)

                    if not trajectory:
                        print("Failed to find a solution!")
                    else:
                        # execute the planned trajectory
                        self.ada.execute_trajectory(trajectory)
                    
                    # lower gripper
                    traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., self.lowerDistance[chosen_obj]])
                    self.ada.execute_trajectory(traj)

                    # ungrab
                    self.hand.ungrab()
                    self.hand.execute_preshape([0.15, 0.15])
                    self.world.remove_skeleton(obj)
                    time.sleep(1)

                    # raise gripper
                    traj = self.ada.plan_to_offset("j2n6s200_hand_base", [0., 0., 0.04])
                    if traj:
                        self.ada.execute_trajectory(traj)

                # ------------------- Move robot back to home ------------------- #
                waypoints = [(0.0, self.ada.get_arm_positions()), (1.0, self.armHome)]
                traj = self.ada.compute_joint_space_path(waypoints)
                self.ada.execute_trajectory(traj)

        print("Finished executing actions.")


# MAIN
# initialise ros node
rospy.init_node("reactive_assembly")
roscpp_init('reactive_assembly', [])
app = QApplication(sys.argv)
win = AssemblyController()
win.showMaximized()
app.exec_()
try:
    rospy.spin()
except KeyboardInterrupt:
    print ("Shutting down")

input("Press Enter to Quit...")

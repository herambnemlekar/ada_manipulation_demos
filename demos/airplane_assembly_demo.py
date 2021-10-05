#!/usr/bin/env python

import adapy
import rospy
import sys
import time
import pickle
import numpy as np
from adarrt import AdaRRT
from gui import *

# set to False if operating real robot
sim = True

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
    Bw[0, 1] = -0.005
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
    :param partPose: SE(3) transform from world to soda can.
    :param adaHand: ADA hand object
    :returns: A fully initialized TSR.
    """

    # set the part TSR at the part pose
    partTSR = adapy.get_default_TSR()
    partTSR.set_T0_w(partPose)

    # transform the TSR to desired grasping pose
    rot_trans = np.eye(4)
    rot_trans[0:3, 0:3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    partTSR_Tw_e = np.matmul(rot_trans, adaHand.get_endeffector_transform("cylinder"))
    partTSR_Tw_e[2] += 0.05

    # set the transformed TSR
    partTSR.set_Tw_e(partTSR_Tw_e)
    Bw = createBwMatrixforTSR()
    partTSR.set_Bw(Bw)

    return partTSR


def toggleHand(adaHand, displacement):
    adaHand.execute_preshape(displacement)


# ------------------------------------------------------- MAIN ------------------------------------------------------- #

# initialise ros node
rospy.init_node("adapy_assembly")
rate = rospy.Rate(10)

if not rospy.is_shutdown():

    # initialize robot
    ada = adapy.Ada(sim)

    # ------------------------------------------ Create sim environment ---------------------------------------------- #

    # objects in airplane assembly
    wingURDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/abstract_main_wing.urdf"
    wingPose = [0.75, -0.3, 0.15, 0.5, 0.5, 0.5, 0.5]

    storageURDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/storage.urdf"
    storagePose = [0., -0.3, -0.77, 0, 0, 0, 0]

    container1URDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/container_1.urdf"
    container1_1Pose = [0.4, -0.4, 0., 0., 0., 0., 0.]
    container1_2Pose = [-0.4, -0.4, 0., 0., 0., 0., 0.]
    container1_3Pose = [0.55, -0.4, 0., 0., 0., 0., 0.]
    container1_4Pose = [-0.55, -0.4, 0., 0., 0., 0., 0.]
    
    container2URDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/container_2.urdf"
    container2_1Pose = [0.4, -0.1, 0, 0., 0., 0., 0.]
    container2_2Pose = [-0.4, -0.1, 0., 0., 0., 0., 0.]

    container3URDFUri = "package://libada/src/scripts/ada-pick-and-place-demos/urdf_collection/container_3.urdf"
    container3_1Pose = [0.6, -0.1, 0., 0., 0., 0., 0.]
    container3_2Pose = [-0.6, -0.1, 0., 0., 0, 0, 0]

    # initialize sim environment
    world = ada.get_world()
    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    # add parts to sim environment
    wing = world.add_body_from_urdf(wingURDFUri, wingPose)
    storageInWorld = world.add_body_from_urdf(storageURDFUri, storagePose)
    container1_1 = world.add_body_from_urdf(container1URDFUri, container1_1Pose)
    container1_2 = world.add_body_from_urdf(container1URDFUri, container1_2Pose)
    container1_3 = world.add_body_from_urdf(container1URDFUri, container1_3Pose)
    container1_4 = world.add_body_from_urdf(container1URDFUri, container1_4Pose)
    container2_1 = world.add_body_from_urdf(container2URDFUri, container2_1Pose)
    container2_2 = world.add_body_from_urdf(container2URDFUri, container2_2Pose)
    container3_1 = world.add_body_from_urdf(container3URDFUri, container3_1Pose)
    container3_2 = world.add_body_from_urdf(container3URDFUri, container3_2Pose)

    rospy.sleep(3.0)

    # ------------------------------------------------ Get robot config ---------------------------------------------- #

    collision = ada.get_self_collision_constraint()

    arm_skeleton = ada.get_arm_skeleton()
    positions = arm_skeleton.get_positions()
    arm_state_space = ada.get_arm_state_space()
    hand = ada.get_hand()
    hand_node = hand.get_endeffector_body_node()

    viewer.add_frame(hand_node)

    # ------------------------------- Start executor for real robot (not needed for sim) ----------------------------- #

    if not sim:
        ada.start_trajectory_executor() 

    armHome = [-1.57, 3.14, 1.23, -2.19, 1.8, 1.2]
    waypoints = [(0.0, positions), (1.0, armHome)]
    trajectory = ada.compute_joint_space_path(arm_state_space, waypoints)
    ada.execute_trajectory(trajectory)
    toggleHand(hand, [0.5, 0.5])
    time.sleep(3)

    # --------------------------------------------------- MAIN Loop -------------------------------------------------- #

    objects = {"main wing": [wing, wingPose],
               "long bolts": [container1_1, container1_1Pose],
               "short bolts": [container1_2, container1_2Pose],
               "propeller nut": [container1_3, container1_3Pose],
               "tail bolt": [container1_4, container1_4Pose],
               "propellers": [container2_1, container2_1Pose],
               "tool": [container2_2, container2_2Pose],
               "propeller hub": [container3_1, container3_1Pose]}

    # initialize gui
    app = QApplication(sys.argv)

    # loop over all objects
    remaining_objects = objects.keys()
    while remaining_objects:
        win = UserInterface()
        win.set_options(remaining_objects)
    
        while not win.user_choice:
            win.show()
            app.exec_()

        print("Robot will fetch:", win.user_choice)

        obj = objects[win.user_choice][0]
        objPose = objects[win.user_choice][1]
        objPoseMat = [[1.0, 0.0, 0.0, objPose[0]],
                      [0.0, 1.0, 0.0, objPose[1]],
                      [0.0, 0.0, 1.0, objPose[2]],
                      [0.0, 0.0, 0.0, 1.0]]
        objTSR = createTSR(objPoseMat, hand)
        # marker = viewer.add_tsr_marker(objTSR)

        # -------------------------------------------- Collision detection ----------------------------------------------- #

        # collision_free_constraint = ada.set_up_collision_detection(ada.get_arm_state_space(), ada.get_arm_skeleton(),
        #                                                                        [obj])
        # full_collision_constraint = ada.get_full_collision_constraint(ada.get_arm_state_space(),
        #                                                                      ada.get_arm_skeleton(),
        #                                                                      collision_free_constraint)
        # collision = ada.get_self_collision_constraint()


        # ------------------------------------------- Setup IK for grasping ---------------------------------------------- #
        
        ik_sampleable = adapy.create_ik(arm_skeleton, arm_state_space, objTSR, hand_node)
        ik_generator = ik_sampleable.create_sample_generator()
        configurations = []
        samples = 0
        maxSamples = 10
        print("Finding IK configuration...")
        while samples < maxSamples and ik_generator.can_sample():
            goal_state = ik_generator.sample(arm_state_space)
            samples += 1
            if len(goal_state) == 0:
                continue
            configurations.append(goal_state)   

        # ------------------------------------------- Plan path for grasping -------------------------------------------- #
        
        if len(configurations) == 0:
            print("No valid configurations found!")
        else:
            print("IK Configuration found!")

            collision = ada.get_self_collision_constraint()

            # trajectory = None
            # adaRRT = AdaRRT(start_state=np.array(armHome), goal_state=np.array(configuration[0]), step_size=0.05,
            #                         goal_precision=0.1, ada=ada, objects=[storageInWorld])
            # path = adaRRT.build()
            # trajectory = None
            # if path is not None:
            #     waypoints = []
            #     for i, waypoint in enumerate(path):
            #         waypoints.append((0.0 + i, waypoint))
            #     trajectory = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)  # 3
           
            waypoints = [(0.0,armHome),(1.0,configurations[0])]
            trajectory = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)

            # tFile = open("trojectCSV/1.txt","w")
            # pickle.dump(trajectory,tFile)
            # tFile.close()

            # ------------------------------------------ Execute path to grasp object --------------------------------- #

            if not trajectory:
                print("Failed to find a solution!")
            else:
                ada.execute_trajectory(trajectory)
                toggleHand(hand, [1.5, 1.5])
                time.sleep(3)
                hand.grab(obj)

                # ----------------------- Lift up grasped object using Jacobian pseudo-inverse ------------------------ #

                # lift up
                waypoints = []
                for i in range(5):
                    jac = arm_skeleton.get_linear_jacobian(hand_node)
                    full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                    delta_x = np.array([0, 0, 0, 0, 0, -0.05])
                    delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                    q = arm_skeleton.get_positions()
                    upWaypt = q + delta_q
                    waypoints.append((i, upWaypt))
                    ada.set_positions(upWaypt)

                # print("Here.", waypoints)
                traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                ada.execute_trajectory(traj)
                time.sleep(3)

                # --------------- Move grasped object to workbench -------------- #

                # move forward
                waypoints = []
                for i in range(5):
                    jac = arm_skeleton.get_linear_jacobian(hand_node)
                    full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                    delta_x = np.array([0, 0, 0, 0, -0.05, 0])
                    delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                    q = arm_skeleton.get_positions()
                    forWaypt = q + delta_q
                    waypoints.append((i, forWaypt))
                    ada.set_positions(forWaypt)

                traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                ada.execute_trajectory(traj)
                time.sleep(3)

                hand.ungrab(obj)
                toggleHand(hand, [0.15, 0.15])
                world.remove_skeleton(obj)
                time.sleep(1)

                # ------------------- Move robot back to home ------------------- #

                waypoints = [(0.0, arm_skeleton.get_positions()), (1.0, armHome)]
                traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                ada.execute_trajectory(traj)
                time.sleep(2)

        print("Finished executing action.")
        remaining_objects.remove(win.user_choice)
        win.user_choice = None

    # --------- Stop executor for real robot (not needed for sim) ----------- #
    if not sim:
        ada.stop_trajectory_executor()


raw_input("Press Enter to Quit...")

# # -------------------------------------------- Create TSR for grasping ------------------------------------------- #
#
# choosed_objectPose = wingPose
#
# objectPoseMat = [[1.0, 0.0, 0.0, choosed_objectPose[0]],
#                  [0.0, 1.0, 0.0, choosed_objectPose[1]],
#                  [0.0, 0.0, 1.0, choosed_objectPose[2]],
#                  [0.0, 0.0, 0.0, 1.0]]
# objectTSR = createTSR(objectPoseMat, hand)
#
# marker = viewer.add_tsr_marker(objectTSR)
#
# choosed_object2Pose = container2_1Pose
#
# object2PoseMat = [[0.0, 0.0, 1.0, choosed_object2Pose[0]],
#                   [0.0, -1.0, 0.0, choosed_object2Pose[1]],
#                   [1.0, 0.0, 0.0, choosed_object2Pose[2]],
#                   [0.0, 0.0, 0.0, 1.0]]
# object2TSR = createTSR(object2PoseMat, hand)

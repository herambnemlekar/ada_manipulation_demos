#!/usr/bin/env python

import adapy
import rospy
import sys
import time
import pickle
import numpy as np
from adarrt import AdaRRT
from gui import *
from assembly_tasks import *

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
    partTSR_Tw_e[2] += 0.0

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

    # dict of all objects
    objects = {"main wing": [wing, wingPose],
               "long bolts": [container1_1, container1_1Pose],
               "short bolts": [container1_2, container1_2Pose],
               "propeller nut": [container1_3, container1_3Pose],
               "tail bolt": [container1_4, container1_4Pose],
               "propellers": [container2_1, container2_1Pose],
               "tool": [container2_2, container2_2Pose],
               "propeller hub": [container3_1, container3_1Pose],
               "tail wing": [container3_2, container3_2Pose]}

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
  
    # ---------------------------------------------- Anticipation stuff ---------------------------------------------- #

    qf = pickle.load(open("q_values.p", "rb"))
    states = pickle.load(open("states.p", "rb"))

    remaining_user_actions = [0, 1, 2, 3, 4, 5, 6, 7]
    required_objects = [["main wing"],
                        ["tail wing"],
                        ["long bolts"],
                        ["tail bolt"],
                        ["tool"],
                        ["tool"],
                        ["propellers", "propeller hub", "short bolts", "tool"],
                        ["propeller nut"]]
    
    current_state = states[0]
    sensitivity = 0.0

    # --------------------------------------------------- MAIN Loop -------------------------------------------------- #

    # initialize gui
    app = QApplication(sys.argv)

    # loop over all objects
    remaining_objects = objects.keys()

    for step in range(17):
        
        # anticipate user action
        s = states.index(current_state)
        max_action_val = -np.inf
        candidates = []
        applicants = []
        for a in remaining_user_actions:
            p, sp = transition(states[s], a)
            if sp:
                applicants.append(a)
                if qf[s][a] > (1 + sensitivity) * max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif (1 - sensitivity) * max_action_val <= qf[s][a] <= (1 + sensitivity) * max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        anticipated_actions = list(set(candidates))
        available_actions = list(set(applicants))


        # objects required for anticipated actions
        available_objects = []
        suggested_objects = []
        for a in available_actions:
            available_objects += required_objects[a]
            if a in anticipated_actions:
                suggested_objects += required_objects[a]
        available_objects = list(set(available_objects))
        suggested_objects = list(set(suggested_objects))

        win = UserInterface()
        win.set_options(available_objects, suggested_objects)
    
        while not win.act:
            win.show()
            app.exec_()

        if win.user_choice:
            objects_to_deliver = win.user_choice
        else:
            objects_to_deliver = suggested_objects

        for chosen_obj in objects_to_deliver:

            obj = objects[chosen_obj][0]
            objPose = objects[chosen_obj][1]
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
                    time.sleep(2)
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

                    print("Picking up.")
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)

                    # --------------- Move grasped object to workbench -------------- #

                    # move forward
                    # waypoints = []
                    # for i in range(9):
                    #     jac = arm_skeleton.get_linear_jacobian(hand_node)
                    #     full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                    #     delta_x = np.array([0, 0, 0, 0, -0.05, 0])
                    #     delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                    #     q = arm_skeleton.get_positions()
                    #     forWaypt = q + delta_q
                    #     waypoints.append((i, forWaypt))
                    #     ada.set_positions(forWaypt)

                    current_position = arm_skeleton.get_positions()
                    new_position = current_position.copy()
                    new_position[0] -= 0.5
                    waypoints = [(0.0, current_position), (1.0, new_position)]
                    print("Moving forward.")
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)

                    # ----------------------- Lower grasped object using Jacobian pseudo-inverse ------------------------ #

                    # lower
                    waypoints = []
                    for i in range(5):
                        jac = arm_skeleton.get_linear_jacobian(hand_node)
                        full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, 0.05])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = arm_skeleton.get_positions()
                        upWaypt = q + delta_q
                        waypoints.append((i, upWaypt))
                        ada.set_positions(upWaypt)

                    print("Keeping down.")
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)

                    hand.ungrab(obj)
                    toggleHand(hand, [0.15, 0.15])
                    world.remove_skeleton(obj)
                    time.sleep(1)

                    # ------------------- Move robot back to home ------------------- #

                    waypoints = [(0.0, arm_skeleton.get_positions()), (1.0, armHome)]
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)

        print("Finished executing action.")
        remaining_objects = [rem_obj for rem_obj in remaining_objects if rem_obj not in objects_to_deliver]

        match_score = 0
        for a in remaining_user_actions:
            a_score = len(set(objects_to_deliver).intersection(required_objects[a]))
            if a_score >= match_score:
                match_score = a_score
                user_action = a

        remaining_user_actions.remove(user_action)
        _, current_state = transition(current_state, user_action) 

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

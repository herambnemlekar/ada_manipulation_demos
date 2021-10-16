#!/usr/bin/env python

import pdb
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


# ------------------------------------------------------- MAIN ------------------------------------------------------- #

# initialise ros node
rospy.init_node("adapy_assembly")
rate = rospy.Rate(10)

if not rospy.is_shutdown():

class AssemblyController:

    def __init__(self):

        # initialize robot
        ada = adapy.Ada(sim)

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
        self.deliveryRotation["long bolts"] = -1.25
        self.graspPose["short bolts"] = [-0.73155659,  4.31674214,  2.28878164, -2.73375183, -1.42453116,  1.24554766]
        self.deliveryRotation["short bolts"] = 1.25
        self.graspPose["propeller nut"] = [0.49796338, 1.90442473,  3.80338018, 2.63336638,  1.44877,  1.67975607]
        self.deliveryRotation["propeller nut"] = -1.0
        self.graspPose["tail bolt"] = [-0.48175263,  4.46387965,  2.68705579, -2.58115143, -1.7464862,   1.62214487]
        self.deliveryRotation["tail bolt"] = 1.0  
        self.graspPose["propellers"] = [-2.4191907,  3.9942575,  1.29241768,  3.05926906, -0.50726387, -0.52933128]
        self.deliveryRotation["propellers"] = -1.0
        self.graspPose["tool"] = [-0.32843145,  4.02576609,  1.48440087, -2.87877031, -0.79457283,  1.40310179]
        self.deliveryRotation["tool"] = 1.05
        self.graspPose["propeller hub"] = [3.00773842,  4.21352853,  1.98663177, -0.17330897,  1.01156224, -0.46210507]  # [-3.10474485,  4.22540059,  2.02201284, -0.19095178,  1.02488858, -0.28577463]
        self.deliveryRotation["propeller hub"] = -0.5
        self.graspPose["tail wing"] = [3.129024,  1.87404028,  3.40826295,  0.53502216, -1.86749865, -0.99044654]
        self.deliveryRotation["tail wing"] = 0.7
        

        # initialize sim environment
        world = ada.get_world()
        viewer = ada.start_viewer("airplane_assembly_demo", "map")

        # add parts to sim environment
        storageInWorld = world.add_body_from_urdf(storageURDFUri, storagePose)
        container1_1 = world.add_body_from_urdf(container1URDFUri, container1_1Pose)
        container1_2 = world.add_body_from_urdf(container1URDFUri, container1_2Pose)
        container1_3 = world.add_body_from_urdf(container1URDFUri, container1_3Pose)
        container1_4 = world.add_body_from_urdf(container1URDFUri, container1_4Pose)
        container2_1 = world.add_body_from_urdf(container2URDFUri, container2_1Pose)
        container2_2 = world.add_body_from_urdf(container2URDFUri, container2_2Pose)
        container3_1 = world.add_body_from_urdf(container3URDFUri, container3_1Pose)
        # container3_2 = world.add_body_from_urdf(container3URDFUri, container3_2Pose)
        tailWing = world.add_body_from_urdf(tailURDFUri, tailPose)
        # wing = world.add_body_from_urdf(wingURDFUri, wingPose)

        # dict of all objects
        self.objects = {"long bolts": [container1_1, container1_1Pose, container1GraspPose, container1GraspOffset],
                        "short bolts": [container1_2, container1_2Pose, container1GraspPose, container1GraspOffset],
                        "propeller nut": [container1_3, container1_3Pose, container1GraspPose, container1GraspOffset],
                        "tail bolt": [container1_4, container1_4Pose, container1GraspPose, container1GraspOffset],
                        "propellers": [container2_1, container2_1Pose, container2GraspPose, container2GraspOffset],
                        "tool": [container2_2, container2_2Pose, container2GraspPose, container2GraspOffset],
                        "propeller hub": [container3_1, container3_1Pose, container3GraspPose, container3GraspOffset],
                        "tail wing": [tailWing, tailPose, tailGraspPose, tailGraspOffset]}

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
        toggleHand(hand, [0.15, 0.15])
      
        # ---------------------------------------------- Anticipation stuff ---------------------------------------------- #

        qf = pickle.load(open("q_values.p", "rb"))
        states = pickle.load(open("states.p", "rb"))

        # actions in airplane assembly and objects required for each action
        n_actions = 8
        remaining_user_actions = list(range(n_actions))
        action_counts = [1, 1, 4, 1, 4, 1, 4, 1]
        required_objects = [["main wing"],
                            ["tail wing"],
                            ["long bolts"],
                            ["tail bolt"],
                            ["tool"],
                            ["tool"],
                            ["propellers", "propeller hub", "short bolts", "tool"],
                            ["propeller nut"]]
        
        # --------------------------------------------------- MAIN Loop -------------------------------------------------- #

        # loop over all objects
        self.remaining_objects = objects.keys()

        # subscribe to action recognition
        sub_act = rospy.Subscriber("/april_tag_detection", Float64MultiArray, self.callback, queue_size=1)

        
    def callback(self, data):

        # update action sequence
        if data != self.user_sequence:
            self.user_sequence = data
        else:
            continue

        # determine current state based on detected action sequence

        # anticipate user action
        sensitivity = 0.0
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


        # determine objects required for anticipated actions
        available_objects = []
        suggested_objects = []
        for a in available_actions:
            available_objects += required_objects[a]
            if a in anticipated_actions:
                suggested_objects += required_objects[a]
        available_objects = list(set(available_objects))
        suggested_objects = list(set(suggested_objects))

        # initialize GUI interface
        app = QApplication(sys.argv)
        win = UserInterface()
        win.set_options(available_objects, suggested_objects)
    
        # keep GUI running until user input is received
        while not win.act:
            win.show()
            app.exec_()

        if win.user_choice:
            objects_to_deliver = win.user_choice
        else:
            objects_to_deliver = suggested_objects

        for chosen_obj in objects_to_deliver:

            if chosen_obj == "main wing":
                print("Get the part by yourself.")
            else:
                print("Providing the required part.")

                obj = objects[chosen_obj][0]
                objPose = objects[chosen_obj][1]
                objGraspPose = objects[chosen_obj][2]
                objOffset = objects[chosen_obj][3]
                objPoseMat = [objGraspPose[0] + [objPose[0] + objOffset[0]],
                              objGraspPose[1] + [objPose[1] + objOffset[1]],
                              objGraspPose[2] + [objPose[2] + objOffset[2]],
                             [0.0, 0.0, 0.0, 1.0]]
                objTSR = createTSR(objPoseMat, hand)
                # marker = viewer.add_tsr_marker(objTSR)
                # raw_input("Does the marker look good?")

                # ---------------------------------------- Collision detection --------------------------------------- #

                # collision_free_constraint = ada.set_up_collision_detection(ada.get_arm_state_space(), ada.get_arm_skeleton(),
                #                                                                        [obj])
                # full_collision_constraint = ada.get_full_collision_constraint(ada.get_arm_state_space(),
                #                                                                      ada.get_arm_skeleton(),
                #                                                                      collision_free_constraint)
                # collision = ada.get_self_collision_constraint()


                # -------------------------------------- Plan path for grasping -------------------------------------- #
                
                if chosen_obj in graspPose.keys():
                    print("Running hard-coded.")
                    grasp_configuration = graspPose[chosen_obj]
                else:
                    # ------------------------------------- Setup IK for grasping ------------------------------------ #
                    ik_sampleable = adapy.create_ik(arm_skeleton, arm_state_space, objTSR, hand_node)
                    ik_generator = ik_sampleable.create_sample_generator()
                    configurations = []
                    samples = 0
                    maxSamples = 1
                    print("Finding IK configuration...")
                    while samples < maxSamples and ik_generator.can_sample():
                        goal_state = ik_generator.sample(arm_state_space)
                        if len(goal_state) == 0:
                            continue
                        else:
                            samples += 1
                        configurations.append(goal_state)   
                    
                    print("Found new configuration.")
                    grasp_configuration = configurations[0]

                waypoints = [(0.0, armHome),(1.0, grasp_configuration)]
                trajectory = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)

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

                # ------------------------------------------ Execute path to grasp object --------------------------------- #

                if not trajectory:
                    print("Failed to find a solution!")
                else:
                    ada.execute_trajectory(trajectory)
                    
                    # lower the griper to grasp object
                    waypoints = []
                    for i in range(3):
                        jac = arm_skeleton.get_linear_jacobian(hand_node)
                        full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, 0.025])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = arm_skeleton.get_positions()
                        new_q = q + delta_q
                        waypoints.append((i, new_q))
                        ada.set_positions(new_q)

                    print("Picking up.")
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)
                    
                    raw_input("Look good to grasp?")
                    
                    toggleHand(hand, [1.5, 1.5])
                    time.sleep(1.5)
                    hand.grab(obj)

                    # ----------------------- Lift up grasped object using Jacobian pseudo-inverse ------------------------ #

                    # lift up
                    waypoints = []
                    for i in range(6):
                        jac = arm_skeleton.get_linear_jacobian(hand_node)
                        full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, -0.025])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = arm_skeleton.get_positions()
                        new_q = q + delta_q
                        waypoints.append((i, new_q))
                        ada.set_positions(new_q)

                    print("Picking up.")
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)

                    # ----------------------------------- Move grasped object to workbench ------------------------------- #
                    
                    current_position = arm_skeleton.get_positions()
                    new_position = current_position.copy()
                    new_position[0] += deliveryRotation[chosen_obj]
                    waypoints = [(0.0, current_position), (1.0, new_position)]
                    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    ada.execute_trajectory(traj)

                    # ----------------------- Lower grasped object using Jacobian pseudo-inverse ------------------------ #

                    # keep down
                    waypoints = []
                    for i in range(4):
                        jac = arm_skeleton.get_linear_jacobian(hand_node)
                        full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                        delta_x = np.array([0, 0, 0, 0, 0, 0.025])
                        delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                        q = arm_skeleton.get_positions()
                        new_q = q + delta_q
                        waypoints.append((i, new_q))
                        ada.set_positions(new_q)

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
        for _ in range(action_counts[user_action]):
            _, current_state = transition(current_state, user_action) 

    # --------- Stop executor for real robot (not needed for sim) ----------- #
    if not sim:
        ada.stop_trajectory_executor()


raw_input("Press Enter to Quit...")

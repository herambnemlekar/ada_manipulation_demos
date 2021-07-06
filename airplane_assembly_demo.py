#!/usr/bin/env python

import adapy
import rospy
import sys
import time
import numpy as np
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
from adarrt import AdaRRT
import pickle


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


def createTSR(partPose, hand):
    """
    Create the TSR for grasping a soda can.
    :param soda_pose: SE(3) transform from world to soda can.
    :param hand: ADA hand object
    :returns: A fully initialized TSR.
    """

    partTSR = adapy.get_default_TSR()
    partTSR.set_T0_w(partPose)

    rot_trans = np.eye(4)
    rot_trans[0:3, 0:3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])


    partTSR_Tw_e = np.matmul(rot_trans, hand.get_endeffector_transform("cylinder"))
    

    partTSR.set_Tw_e(partTSR_Tw_e)
    Bw = createBwMatrixforTSR()
    partTSR.set_Bw(Bw)

    return partTSR

def closeHand(hand, displacement):
    hand.execute_preshape(displacement)


roscpp_init('airplane_assembly', [])
rospy.init_node("adapy_assembly")
rate = rospy.Rate(10)

if not rospy.is_shutdown():

    sim = True

    ada = adapy.Ada(sim)

    # -------------------------- Create sim world --------------------------- #

    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    obj1URDFUri = "package://pr_assets/data/objects/containerBig.urdf"
    object1Pose = [0.21, 0.44, 0.038, 0.707, 0., 0., 0.707]

    obj2URDFUri = "package://pr_assets/data/objects/containerBig.urdf"
    object2Pose = [0.00, 0.44, 0.038, 0.707, 0., 0., 0.707]  # [0.00, -0.5, 0.012, 0.707, 0., 0., 0.707]

    obj3URDFUri = "package://pr_assets/data/objects/containerMid.urdf"
    object3Pose = [-0.14, 0.42, 0.032, 0.707, 0., 0., 0.707]

    obj4URDFUri = "package://pr_assets/data/objects/containerMid.urdf"
    object4Pose = [-0.24, 0.44, 0.032, 0.707, 0., 0., 0.707]

    obj5URDFUri = "package://pr_assets/data/objects/containerMid.urdf"
    object5Pose = [0.00, 0.62, 0.095, 0.707, 0., 0., 0.707]

    obj6URDFUri = "package://pr_assets/data/objects/containerMid.urdf"
    object6Pose = [-0.10, 0.58, 0.095, 0.707, 0., 0., 0.707]

    world = ada.get_world()

    obj1 = world.add_body_from_urdf(obj1URDFUri, object1Pose)
    obj2 = world.add_body_from_urdf(obj2URDFUri, object2Pose)
    obj3 = world.add_body_from_urdf(obj3URDFUri, object3Pose)
    obj4 = world.add_body_from_urdf(obj4URDFUri, object4Pose)

    rospy.sleep(5.0)

    # -------------------------- Get robot config --------------------------- #

    collision = ada.get_self_collision_constraint()

    arm_skeleton = ada.get_arm_skeleton()
    positions = arm_skeleton.get_positions()
    arm_state_space = ada.get_arm_state_space()
    hand = ada.get_hand()
    hand_node = hand.get_endeffector_body_node()

    viewer.add_frame(hand_node)

    # --------- Start executor for real robot (not needed for sim) ---------- #

    if not sim:
        ada.start_trajectory_executor()

    # ----------------------- Create TSR for grasping ----------------------- #


    choosed_objectPose = object6Pose

    objectPoseMat = [[1.0, 0.0, 0.0, choosed_objectPose[0]],
                      [0.0, 1.0, 0.0, choosed_objectPose[1]],
                      [0.0, 0.0, 1.0, choosed_objectPose[2]],
                      [0.0, 0.0, 0.0, 1.0]]
    objectTSR = createTSR(objectPoseMat,hand)


    marker = viewer.add_tsr_marker(objectTSR)

     # ------------------ Move robot to start configuration ----------------- #

    #target waypoint: [-2.77053788,  4.20495852,  1.98182475, -3.38792973,  0.04354109, 0.33316412]
    #armHome = [-1.5094078 ,  2.92774907,  1.08108148, -1.30679823,  1.72727102, 2.50344173]

    armHome = [-1.57, 3.14, 1.23, -2.19, 1.8, 1.2]
    print("Moving robot to home location...")
    waypoints = [(0.0, positions), (1.0, armHome)]
    trajectory = ada.compute_joint_space_path(arm_state_space, waypoints)
    
    ada.execute_trajectory(trajectory)
    closeHand(hand, [0.5, 0.5])
    time.sleep(3)

    # ------------------------ Setup IK for grasping ------------------------ #
    
    ik_sampleable = adapy.create_ik(arm_skeleton, arm_state_space, objectTSR, hand_node)
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

    # ------------------------ Plan path for grasping ----------------------- #
    
    if len(configurations) == 0:
        print("No valid configurations found!")
    else:
        print("IK Configuration found!")

        #ada.set_arm_positions(configurations[0])

        collision_free_constraint = ada.set_up_collision_detection(ada.get_arm_state_space(),
                                                                            ada.get_arm_skeleton(),
                                                                            [obj1,obj2,obj3,obj4])
        full_collision_constraint = ada.get_full_collision_constraint(ada.get_arm_state_space(),
                                                                              ada.get_arm_skeleton(),
                                                                              collision_free_constraint)
        collision = ada.get_self_collision_constraint()

        # trajectory = None
        # for configuration in configurations:
        #     adaRRT = AdaRRT(start_state=np.array(armHome), goal_state=np.array(configuration), step_size=0.01,
        #                             goal_precision=0.2, ada=ada, objects=[obj1])
        #     path = adaRRT.build()
        #     trajectory = None
        #     if path is not None:
        #         waypoints = []
        #         for i, waypoint in enumerate(path):
        #             waypoints.append((0.0 + i, waypoint))
        #         trajectory = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)  # 3
       
        waypoints = [(0.0,armHome),(1.0,configurations[0])]
        trajectory = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)

        #tFile = open("trojectCSV/1.txt","w")
        #pickle.dump(trajectory,tFile)
        #tFile.close()

        # ------------------ Execute path to grasp object ------------------- #

        if not trajectory:
            print("Failed to find a solution!")
        else:
            print("Found a trajectory! Executing...")

            ada.execute_trajectory(trajectory)

            closeHand(hand, [1.5, 1.5])

            time.sleep(4)

            #closeHand(hand, [0.3, 0.3])


            # next step transfer Jacobian pseudo-inverse for forward motion
            '''
            step_size = 0.01
            num_steps = 4
            for ii in range(0, num_steps):
                jac = arm_skeleton.get_linear_jacobian(hand_node)
                full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
                delta_x = np.array([0, 0, 0, 0, 0, -step_size])
                delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
                q = arm_skeleton.get_positions()
                next_q = q + delta_q
                if sim:
                    ada.set_positions(next_q)
                    viewer.update()
                else:
                    print "next_q", next_q
                    waypoints = [(0.0, q), (1.0, next_q)]
                    trajectory = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
                    # ada.start_trajectory_executor()
                    ada.execute_trajectory(trajectory)
                    # time.sleep(0.01)
			'''

            # ------------------- Lift up grasped object -------------------- #

            #lift up
            jac = arm_skeleton.get_linear_jacobian(hand_node)
            full_jac = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
            delta_x = np.array([0, 0, 0, 0, 0, -0.3])
            delta_q = np.matmul(np.linalg.pinv(full_jac), delta_x)
            q = arm_skeleton.get_positions()
            upWaypt = q + delta_q


            waypoints = [(0.0,configurations[0]),(1.0,upWaypt)]
            traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
            ada.execute_trajectory(traj)
            time.sleep(3)

            # --------------- Move grasped object to workbench -------------- #

            targetWaypt = [-3.41037512,  2.48071804,  4.96467289, -3.48166341,  0.68219961, -2.10886992]
            waypoints = [(0.0,upWaypt),(1.0,targetWaypt)]
            traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
            ada.execute_trajectory(traj)
            time.sleep(3)
            closeHand(hand, [0.3, 0.3])

            # ------------------- Move robot back to home ------------------- #

            waypoints = [(0.0,targetWaypt),(1.0,armHome)]
            traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
            ada.execute_trajectory(traj)
            time.sleep(2)

    # --------- Stop executor for real robot (not needed for sim) ----------- #
    if not sim:
        ada.stop_trajectory_executor()


raw_input("Press Enter to Quit...")

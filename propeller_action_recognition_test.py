#!/usr/bin/env python3

import sys
import time
import os
from sys import platform
from os import write
import cv2
import apriltag
import time
import csv
from collections import defaultdict
import copy

import rospy
from std_msgs.msg import *

import pickle
import numpy as np


starttime_long_bolts = time.time()
starttime_short_bolts = time.time()
starttime_propeller_blades = time.time()
starttime_tool = time.time()
# ---------------------------------------- Skeleton Tracking ----------------------------------------- #

# Import Openpose (Windows/Ubuntu/OSX)
# dir_path = os.path.dirname(os.path.realpath(__file__))
# try:
#     # Windows Import
#     if platform == "win32":
#         # Change these variables to point to the correct folder (Release/x64 etc.)
#         sys.path.append(dir_path + '/../../python/openpose/Release');
#         os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
#         import pyopenpose as op
#     else:
#         # Change these variables to point to the correct folder (Release/x64 etc.)
#         #sys.path.append('../../python');
#         # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
#         sys.path.append('/usr/local/python')
#         from openpose import pyopenpose as op
        

# except ImportError as e:
#     print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
#     raise e


# # Custom Params (refer to include/openpose/flags.hpp for more parameters)
# params = dict()

# #params["model_folder"] = "../../../models/"
# params["model_folder"] = "/home/suraj/src/openpose/models/"
# params["face"] = False
# params["hand"] = True


# # Starting OpenPose
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()

# starttime = time.time()

# structTime = time.localtime(starttime)

# nowTime = time.strftime("%Y-%m-%d %H:%M:%S", structTime)

# csv_path = "/home/suraj/openPose_record/"+str(nowTime)+".csv"

# f = open(csv_path, 'w')

# writer = csv.writer(f)

# writer.writerow(["timeStamp","Body_raw","Left_Hand_raw", "Right_Hand_raw"])

# f.flush()

# global variables for detecting propeller blades
propeller_done = False
detect_propeller_action = False
last_propeller_state = False
propeller_off_counter = 0.0
propeller_count = 0

# global variables for detecting long bolts
last_long_bolt_state = False
long_bolt_counter = 0
long_bolt_off_counter = 0.0
insert_long_bolt_count = 0
screw_long_bolt_count = 0
took_all_long_bolt_left = False

# global variables for detecting short bolts
detect_short_bolt_action = False
last_short_bolt_state = False
short_bolt_counter = 0
short_bolt_off_counter = 0.0
took_all_short_bolt_left = False

screw_propeller_action_count = 0

detect_tool = False
detect_empty_tool = False
last_empty_tool = False 
tool_off_counter = 0.0
#detect_screw_action = False
# --------------------------------------- Action Recognition ----------------------------------------- #

txt_dir = "/home/suraj/aprilTagVideo/aprilTag_state/"


class Action:
    def __init__(self, id = -1, name='', parts_combo = []):
       self.id = id
       self.parts_combo_list = [set(combo) for combo in parts_combo]
       self.name = name

    def has_parts(self, part_set):
        for combo in self.parts_combo_list:
            if combo <= part_set:
                return True


# Requires all AprilTag for all parts (except for the tool and long screws) 
# actions_list = [Action(1, 'Insert main wing into body', [[0, 1]]), #near tag 1
#                Action(2, 'Screw main wing to body', [[0, 1, 21, 17, 18]), #near tag 1
#                Action(3, 'Insert right wing tip into main wing', [[1, 4]]),
#                Action(4, 'Insert left wing tip into main wing', [[1, 3]]),
#                Action(5, 'Insert tail wing into body', [[0, 2]]),
#                Action(6, 'Screw tail wing to body', [[0, 2, 13, 17], [0, 2, 21, 17], [0, 2, 13, 18], [0, 2, 21, 18]]),
#                Action(7, 'Screw propeller to propeller base', [[5, 6, 19, 20, 14, 22, 17], [5, 6, 19, 20, 14, 22, 18]]),
#                Action(8, 'Screw propeller base to body', [[0, 5, 6, 16, 24]]),
#                Action(9, 'Screw propeller cap to propeller base', [[0, 5, 6, 7, 8, 15, 23, 17], [0, 5, 6, 7, 8, 15, 23, 17]]),]

# actions_list = [Action([0], 'Insert main wing into body', [[0, 1]]), #near tag 1
#                Action([2, 4], 'Screw main wing to body', [[0, 1, 21, 17, 18]]), #near tag 1
#                Action([1], 'Insert tail wing into body', [[0, 2]]), #near tag 2
#                Action([3, 5], 'Screw tail wing to body', [[0, 2, 30, 17, 18]]), #near tag 2
#                Action([6], 'Screw propeller to propeller base', [[5, 6, 19, 20, 14, 22, 17,18]]),
#                Action([7], 'Screw propeller base to body', [[0, 5, 6, 16, 24]]),
#                ]

               #13 shown: action 2 and action 6 finished

parts_list = {"21": "long bolts",
              "22": "short bolts",
              "24": "propeller nut",
              "30": "tail screw",
              "20": "propeller blades",
              "17": "tool",
              "5": "propeller hub",
              "2": "tail wing",
              "1": "main wing",
              "0": "airplane body"}

# actions_list = [Action([0], 'Insert main wing', [[0, 1]]), #near tag 1
#                Action([2, 4], 'Screw main wing', [[0, 1, 21, 17, 18]]), #near tag 1
#                Action([1], 'Insert tail wing', [[0, 2]]), #near tag 2
#                Action([3, 5], 'Screw tail wing', [[0, 2, 30, 17, 18]]), #near tag 2
#                Action([6], 'Screw propellers', [[5, 6, 19, 20, 14, 22, 17,18]]),
#                Action([7], 'Fix propeller hub', [[0, 5, 6, 16, 24]]),
#                ]

# //EDIT: add action: Screw one propeller & Screw all propeller; delete part 20 from combo
# if 20 is detected for a while then it means one propeller has been taken. it will be gone because the robot would take it back
# 19 is probably empty propeller blade container

actions_list = [Action([0], 'Insert main wing', [[0, 1]]), #near tag 1
               Action([2], 'Insert bolt in main wing', [[0, 1]]), #near tag 1
               Action([4], 'Screw bolt to main wing', [[0, 1, 27, 17]]), #near tag 1
               Action([1], 'Insert tail wing', [[0, 2]]), #near tag 2
               Action([3], 'Insert bolt in tail wing', [[0, 2, 30]]), #near tag 2
               Action([5], 'Screw bolt to tail wing', [[0, 2, 30, 17, 9]]), #near tag 2
               Action([6], 'Screw one propellers', [[5, 17]]), #x4 # 17: empty tool
               Action([7], 'Fix propeller hub', [[0, 5, 16, 24]])
               ]


actions_from_part = defaultdict(set)

for action in actions_list:
    for combo in action.parts_combo_list:
        for part in combo:
            actions_from_part[part].add(action)

performed_actions = [0] * len(actions_list)
undone_actions = copy.deepcopy(actions_list)
part_set = set()


def detect_apriltag(gray, image, state):
    global starttime_long_bolts, starttime_short_bolts, starttime_propeller_blades, starttime_tool
    global performed_actions, part_set, action_sequence
    global propeller_done, detect_propeller_action, last_propeller_state, propeller_off_counter, propeller_count, screw_propeller_action_count
    global last_long_bolt_state, long_bolt_counter, long_bolt_off_counter, insert_long_bolt_count, screw_long_bolt_count, took_all_long_bolt_left
    global detect_short_bolt_action, last_short_bolt_state, short_bolt_counter, short_bolt_off_counter, took_all_short_bolt_left
    global detect_tool, detect_empty_tool, last_empty_tool, tool_off_counter
    
    ifRecord = False

    #for linux
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    # results = detector.detect(img=gray,True, camera_params=[544.021136,542.307110,308.111905,261.603373], tag_size=0.044)
    results = detector.detect(img=gray)

    # if len(results) > 0:
    #     # print("[INFO] {} total AprilTags detected".format(len(results)))
    #     useless = 0
    # else:
    #     # print("No AprilTag Detected")
    #     return image, ifRecord

    # loop over the AprilTag detection results

    # variables for detecting if propeller blades and bolts are detected by camera right now
    # short bolts container: 22
    # long bolts container: 21
    propeller_detected = False
    long_bolts_detected = False
    short_bolts_detected = False
    detect_tool = False 
    detect_empty_tool = False

    # ids = []
    # for r in results:
    #     ids.append(r.tag_id)
    # print(ids)

    #//Edit: Need to add special process for tag 19 (empty propeller blade box) and 20(propeller blade)
    for r in results:
        # if detected propeller
        if r.tag_id == 20:
            propeller_detected = True
            #part_set.add(r.tag_id)

        elif r.tag_id == 21:
            long_bolts_detected = True
            #part_set.add(r.tag_id)

        elif r.tag_id == 22:
            short_bolts_detected = True
            #part_set.add(r.tag_id)

        elif r.tag_id == 19:
            part_set.add(r.tag_id)
            part_set.add(20)
            propeller_count = 4
            #print("add propeller blades to the list")

        elif r.tag_id == 18:
            # tool box
            # detect_tool = True 
            if not 18 in part_set:
                part_set.add(18)

        elif r.tag_id == 17:
            # tool empty tag
            detect_tool = True 
            detect_empty_tool = True
            if not 17 in part_set:
                part_set.add(17)
            
        elif r.tag_id == 25:
            took_all_long_bolt_left = True 

        elif r.tag_id == 26:
            took_all_short_bolt_left = True 

        elif r.tag_id == 27:
            if insert_long_bolt_count > screw_long_bolt_count and not 27 in part_set:
                part_set.add(27)

        # AprilTag state
        elif r.tag_id > 32:
            #print("tag id:",r.tag_id)
            continue

        elif state[r.tag_id] == 0:
            ifRecord = True
            # Action Detection
            part_set.add(r.tag_id)
    
        state[r.tag_id] = 1

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        # cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")

        # print("[INFO] tag family: {}".format(tagFamily))
        # M, e1, e2 = detector.detection_pose(r, [544.021136, 542.307110, 308.111905, 261.603373])
        # w:QR code length
        # w = 4.4
        # t = [M[0][3] * w, M[1][3] * w, M[2][3] * w]

        # dist = (t[0] ** 2 + t[1] ** 2 + t[2] ** 2) ** 0.5

        showStr = "ID: "+str(r.tag_id)

        cv2.putText(image, showStr, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # updated global variables with a timer
    if not propeller_done:
        # if propeller_detected and last_propeller_state == False and propeller_off_counter >= 3.0:
        #     detect_propeller_action = True 
        if propeller_detected == False and last_propeller_state == True:
            starttime_propeller_blades = time.time()
            if propeller_detected < 4:
                propeller_count += 1
        if propeller_detected == False and last_propeller_state == False:
            propeller_off_counter = time.time() - starttime_propeller_blades

    # increase count when first see the object
    if long_bolt_counter<5 and not took_all_long_bolt_left:
        if long_bolts_detected and last_long_bolt_state == False and long_bolt_off_counter >= 3.0: 
            long_bolt_counter += 1
            long_bolt_off_counter = 0.0
            print("increment counter long",long_bolt_counter)
            if long_bolt_counter == 4 and not (21 in part_set):
                part_set.add(21)
                print("add long bolt to the list")
        if long_bolts_detected == False and last_long_bolt_state == True:
            starttime_long_bolts = time.time()
        if long_bolts_detected == False and last_long_bolt_state == False:
            long_bolt_off_counter = time.time() - starttime_long_bolts
    elif long_bolt_counter < 4 and took_all_long_bolt_left and not 21 in part_set:
        part_set.add(21)


    if short_bolt_counter<5 and not took_all_short_bolt_left:
        if short_bolts_detected and last_short_bolt_state == False and short_bolt_off_counter >= 3.0:
            detect_short_bolt_action = True
            short_bolt_counter += 1
            if short_bolt_counter == 4 and not (22 in part_set):
                took_all_short_bolt_left = True
                print("add all short bolts")
                part_set.add(22)
        if short_bolts_detected == False and last_short_bolt_state == True:
            starttime_short_bolts = time.time()
        if short_bolts_detected == False and last_short_bolt_state == False:
            short_bolt_off_counter = time.time() - starttime_short_bolts
    elif short_bolt_counter < 4 and took_all_short_bolt_left and not 22 in part_set:
        print("add all short bolts")
        short_bolt_counter = 4
        part_set.add(22)
    
    # tool
    # if detect_empty_tool == True and last_empty_tool == False and tool_off_counter >=3.0:
    #     detect_screw_action = True
    if detect_empty_tool == False and last_empty_tool == True:
        starttime_tool = time.time()
    if detect_empty_tool == False and last_empty_tool == False:
        tool_off_counter = time.time() - starttime_tool
    if detect_empty_tool == False and detect_tool == False and tool_off_counter >= 3.0:
        if 18 in part_set:
            part_set.remove(18)
        if 17 in part_set:
            part_set.remove(17)

    last_empty_tool = detect_empty_tool
    last_short_bolt_state = short_bolts_detected
    last_long_bolt_state = long_bolts_detected
    last_propeller_state = propeller_detected

    # print("[INFO] dist:",dist," tag pose:",t)
    return image, ifRecord


def video_demo():
    global propeller_done, detect_propeller_action, last_propeller_state, propeller_count, screw_propeller_action_count
    global last_long_bolt_state, long_bolt_counter, insert_long_bolt_count, screw_long_bolt_count, took_all_long_bolt_left
    global detect_short_bolt_action, last_short_bolt_state, short_bolt_counter, took_all_short_bolt_left
    global detect_tool, detect_empty_tool, last_empty_tool, tool_off_counter

    # publisher
    ros_pub = rospy.Publisher("/april_tag_detection", Float64MultiArray, queue_size=1)

    state = [0 for _ in range(32)]

    ground_truth_action_sequence = [1,7,8,2,5,6]

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # capture.set(cv2.CAP_PROP_POS_FRAMES,6000)

    # ret, mtx, dist, rvecs, tvecs = pickle.load(open('/home/suraj/camera_parameters/webcam_rough_calibration.pickle','rb'))
    # w = 1920
    # h = 1080
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # x, y, w, h = roi

    count = 0
    action_sequence = []
    legible_action_sequence = " "

    while (True):
        
        ref, frame = capture.read()

        image = frame

        # action recognition
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image, ifRecord = detect_apriltag(gray, image.copy(), state)

        for action in undone_actions:
            if action.id[0] == 6:
                # special processing for action 6
                if screw_propeller_action_count >= 4:
                    propeller_done = True
                    undone_actions.remove(action)
                    print("Propeller Done")
                else:
                    if not propeller_done:
                        if action.has_parts(part_set):
                            #print(short_bolt_counter, propeller_count, screw_propeller_action_count)
                            while min(short_bolt_counter, propeller_count) > screw_propeller_action_count:
                                action_sequence += action.id
                                legible_action_sequence += action.name + ", "
                                screw_propeller_action_count += 1
                                print("screw propeller count: ", screw_propeller_action_count)
            elif action.id[0] == 2:
                if long_bolt_counter < 5 and took_all_long_bolt_left:
                    long_bolt_counter = 4
                    if action.has_parts(part_set):
                        while insert_long_bolt_count < long_bolt_counter:
                            action_sequence += action.id
                            legible_action_sequence += action.name + ", "
                            print("Add Insert Long bolt actions")
                            insert_long_bolt_count += 1
                elif long_bolt_counter < 5 and not took_all_long_bolt_left:
                    if action.has_parts(part_set):
                        while insert_long_bolt_count < long_bolt_counter:
                            action_sequence += action.id
                            legible_action_sequence += action.name + ", "
                            print("Add Insert Long bolt actions")
                            insert_long_bolt_count += 1
                if insert_long_bolt_count == 4:
                    print("Insert Long bolt done")
                    undone_actions.remove(action)

            elif action.id[0] == 4:
                # screw long bolt
                if action.has_parts(part_set):
                    if long_bolt_counter < 5 and insert_long_bolt_count > screw_long_bolt_count:
                        action_sequence += action.id
                        legible_action_sequence += action.name + ", "
                        screw_long_bolt_count += 1
                        print("Add Screw Long bolt actions")
                        part_set.remove(27)
                    if screw_long_bolt_count == 4:
                        undone_actions.remove(action)
                        print("Screw Long bolt done")
            elif action.has_parts(part_set):
                #print(action.id)
                action_sequence += action.id
                legible_action_sequence += action.name + ", "
                undone_actions.remove(action)
       
        legible_part_list = ""
        for part_id in part_set:
            if str(part_id) in parts_list.keys():
                legible_part_list += parts_list[str(part_id)] + ","
        legible_part_list = legible_part_list[:-1]
        #print(legible_part_list)


        cv2.rectangle(image, (0, 0), (1920, 100), (0,0,0), -1)
        cv2.putText(image, "detected tags: " + legible_part_list, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        display_text = "Detected Action Sequence: " + legible_action_sequence
        line_length = 160
        start_index = 0
        start_height = 75
        while start_index < len(display_text):
            end_index =  len(display_text) if start_index+line_length > len(display_text) else start_index+line_length
            cv2.putText(image, display_text[start_index:end_index], (5, start_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            start_index += line_length
            start_height += 35
        cv2.imshow('AprilTag', image)

        tag_info = MultiArrayDimension()
        tag_info.label = legible_part_list
        tag_layout = MultiArrayLayout()
        tag_layout.dim = [tag_info]

        seq = Float64MultiArray()
        seq.layout = tag_layout
        seq.data = np.array(action_sequence)
        ros_pub.publish(seq)
        
        c = cv2.waitKey(100)

        if c == 27:
            capture.release()
            f.close()
            break

        count+=1

rospy.init_node('april_tag_detection', anonymous=True)
video_demo()

try:
    rospy.spin()
except KeyboardInterrupt:
    print ("Shutting down")
cv2.destroyAllWindows()

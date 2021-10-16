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
from std_msgs.msg import Float64MultiArray

import pickle
import numpy as np

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


# action recognition
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

actions_list = [Action([0], 'Insert main wing into body', [[0, 1]]), #near tag 1
               Action([2, 4], 'Screw main wing to body', [[0, 1, 21, 17, 18]]), #near tag 1
               Action([1], 'Insert tail wing into body', [[0, 2]]), #near tag 2
               Action([3, 5], 'Screw tail wing to body', [[0, 2, 21, 17, 18]]), #near tag 2
               Action([6], 'Screw propeller to propeller base', [[5, 6, 19, 20, 14, 22, 17,18]]),
               Action([7], 'Screw propeller base to body', [[0, 5, 6, 16, 24]]),
               ]

               #13 shown: action 2 and action 6 finished


actions_from_part = defaultdict(set)

for action in actions_list:
    for combo in action.parts_combo_list:
        for part in combo:
            actions_from_part[part].add(action)

performed_actions = [0] * len(actions_list)

undone_actions = copy.deepcopy(actions_list)

# action_sequence = []

part_set = set()

def detect_apriltag(gray, image, state):
    global performed_actions, part_set, action_sequence
    
    ifRecord = False

    #for linux
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    # results = detector.detect(img=gray,True, camera_params=[544.021136,542.307110,308.111905,261.603373], tag_size=0.044)
    results = detector.detect(img=gray)

    if len(results) > 0:
        # print("[INFO] {} total AprilTags detected".format(len(results)))
        useless = 0
    else:
        # print("No AprilTag Detected")
        return image, ifRecord

    # loop over the AprilTag detection results
    for r in results:
        # AprilTag state
        if r.tag_id > 24:
            print("tag id:",r.tag_id)
            continue

        if state[r.tag_id] == 0:
            ifRecord = True
            # Action Detection
            part_set.add(r.tag_id)

            

        state[r.tag_id] = 1
        #print(state)

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

        # showStr = "dist:" + str(dist)
        showStr = "ID: "+str(r.tag_id)

        cv2.putText(image, showStr, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # print("[INFO] dist:",dist," tag pose:",t)

    return image, ifRecord


def video_demo():


    # publisher
    ros_pub = rospy.Publisher("/april_tag_detection", Float64MultiArray, queue_size=1)

    state = [0 for _ in range(30)]

    video_name = "good assembly example"
    ground_truth_action_sequence = [1,7,8,2,5,6]

    #capture = cv2.VideoCapture("../../aprilTagVideo/user_study_video/" + video_name + ".webm")

    # capture = cv2.VideoCapture("/home/suraj/good assembly example.mkv")
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # capture.set(cv2.CAP_PROP_POS_FRAMES,6000)

    # ret, mtx, dist, rvecs, tvecs = pickle.load(open('/home/suraj/camera_parameters/webcam_rough_calibration.pickle','rb'))
    # w = 1920
    # h = 1080
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # x, y, w, h = roi

    with open(txt_dir + 'AprilTag_State_' + video_name + '.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerow(['frameInd','timeStamp']+[str(x) for x in range(0,30)]+["performed actions"])
        f.flush()

        count = 0
        action_sequence = []

        while (True):
            ref, frame = capture.read()

            # dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # # # dst = dst[y:y+h, x:x+w]
            # frame = dst

            image = frame

            # # Process Image
            # datum = op.Datum()
            # imageToProcess = image
            # datum.cvInputData = imageToProcess
            # opWrapper.emplaceAndPop(op.VectorDatum([datum]))


            # #Display Image
            # # print("Body keypoints: \n" + str(datum.poseKeypoints))
            # # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            # # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

            # jObject = {}

            # jObject["poseKeypoints"] = None
            # jObject["leftHandKeypoints"] = None
            # jObject["rightHandKeypoints"] = None
            
            # x_max = 0
            # y_max = 0

            # #pose
            # if not (datum.poseKeypoints is None):

            #     jObject["poseKeypoints"] = datum.poseKeypoints.tolist()


            #     for j in range(0,len(datum.poseKeypoints)):

            #         pk_list_1 = []

            #         body_kps = datum.poseKeypoints[j]

            #         for i in range(0,len(body_kps)):
            #             x = body_kps[i][0]
            #             y = body_kps[i][1]
            #             confidence = body_kps[i][2]

            #             #print("x:",x)
            #             #print("y:",y)

            #             x_max = max(x_max,x)
            #             y_max = max(y_max,y)

            #             radius = 5

            #             image = cv2.circle(image,(x,y),radius,(0,255,0),-1)

            
            # # left hand
            # if not (datum.handKeypoints[0] is None):

            #     jObject["leftHandKeypoints"] = datum.handKeypoints[0].tolist()


            #     for j in range(0,len(datum.handKeypoints[0])):

            #         left_kps = datum.handKeypoints[0][j]

            #         for i in range(0,len(left_kps)):
            #             x = left_kps[i][0]
            #             y = left_kps[i][1]
            #             confidence = left_kps[i][2]

            #             # print("x:",x)
            #             # print("y:",y)

            #             x_max = max(x_max,x)
            #             y_max = max(y_max,y)

                        
            #             radius = 2

            #             image = cv2.circle(image,(x,y),radius,(0,0,255),-1) 

            
            # # right hand
            # if not (datum.handKeypoints[1] is None):

            #     jObject["rightHandKeypoints"] = datum.handKeypoints[1].tolist()

            #     for j in range(0,len(datum.handKeypoints[1])):

            #         right_kps = datum.handKeypoints[1][j]

            #         for i in range(0,len(right_kps)):
            #             x = right_kps[i][0]
            #             y = right_kps[i][1]
            #             confidence = right_kps[i][2]

            #             # print("x:",x)
            #             # print("y:",y)

            #             x_max = max(x_max,x)
            #             y_max = max(y_max,y)


            #             radius = 2

            #             image = cv2.circle(image,(x,y),radius,(255,0,0),-1) 
    


            # curTime = time.time()

            # writer.writerow([curTime,jObject["poseKeypoints"],jObject["leftHandKeypoints"],jObject["rightHandKeypoints"]])
            
            # f.flush()

            # print("max_x",x_max)
            # print("max_y",y_max)

            # if curTime-starttime > 2.0:
            #     #cv2.imwrite("sample_img/"+str(curTime)+".jpg",datum.cvOutputData)
            #     starttime = curTime

               

            # action recognition
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image, ifRecord = detect_apriltag(gray, image.copy(), state)

            if ifRecord:
                global performed_actions
                writer.writerow([count, time.time()] + state + action_sequence)
                print("state change!!")
                f.flush()

            for action in undone_actions:
                if action.has_parts(part_set):
                    action_sequence += action.id
                    undone_actions.remove(action)

            cv2.rectangle(image, (0, 0), (500, 125), (0,0,0), -1)
            cv2.putText(image, "Ground Truth Action Sequence: " + str(ground_truth_action_sequence), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "Detected Action Sequence: " + str(action_sequence), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(image, "detected tags: " + str(part_set), (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "undone actions: " + str([x.id for x in undone_actions]), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('AprilTag', image)

            seq = Float64MultiArray()
            seq.data = np.array(action_sequence)
            ros_pub.publish(seq)
            
            c = cv2.waitKey(1)

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

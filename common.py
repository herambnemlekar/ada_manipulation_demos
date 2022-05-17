from copy import deepcopy
import numpy as np
import adapy



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

def createTSR(partPose, graspPose):
    """
    Create the TSR for grasping a soda can.
    :param partPose: SE(3) transform from world to part.
    :param adaHand: ADA hand object
    :returns: A fully initialized TSR.  
    """

    # set the part TSR at the part pose
    partTSR = adapy.get_default_TSR()
    T0_w = partTSR.get_T0_w()
    T0_w[0:3, 3] = partPose[:3]
    partTSR.set_T0_w(T0_w)

    # set the transformed TSR
    partTSR.set_Tw_e(graspPose)
    Bw = createBwMatrixforTSR()
    partTSR.set_Bw(Bw)

    return partTSR

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
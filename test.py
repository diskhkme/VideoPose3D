import numpy as np
import matplotlib.pyplot as plt
from time import sleep

keypoints = np.load('data/data_2d_h36m_cpn_ft_h36m_dbb.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()
subject_keypoints = keypoints['S6']
test_keypoints = subject_keypoints['Directions 1'][0]

# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/data_utils.py
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip' # 8
H36M_NAMES[1]  = 'RHip' # 12
H36M_NAMES[2]  = 'RKnee' # 13
H36M_NAMES[3]  = 'RFoot' # 14
H36M_NAMES[4]  = 'LHip' # 9
H36M_NAMES[5]  = 'LKnee' # 10
H36M_NAMES[6]  = 'LFoot' # 11
H36M_NAMES[7] = 'Spine' # ??
H36M_NAMES[8] = 'Thorax' # 1
H36M_NAMES[9] = 'Neck/Nose' # 0
H36M_NAMES[10] = 'Head' # ??
H36M_NAMES[11] = 'LShoulder' # 2
H36M_NAMES[12] = 'LElbow' # 3
H36M_NAMES[13] = 'LWrist' # 4
H36M_NAMES[14] = 'RShoulder' # 5
H36M_NAMES[15] = 'RElbow' # 6
H36M_NAMES[16] = 'RWrist' # 7

child_to_parant_dict = {1:0, 2:1, 3:2, 4:0, 5:4, 6:5, 7:0, 8:7, 9:8, 10:9,
                   11:8, 12:11, 13:12, 14:8, 15:14, 16:15}


frameNum = 0
while frameNum < test_keypoints.shape[0]:
    plt.cla()
    plt.axis([0, 1024, 0, 1024])
    plt.gca().invert_yaxis()
    joint2D_data = test_keypoints[frameNum,:,:]
    #plt.scatter(joint2D_data[:,0], joint2D_data[:,1])
    for i in range(17):
        if i != 0:
            parent_index = child_to_parant_dict[i]
            plt.plot((joint2D_data[i,0], joint2D_data[parent_index,0]), (joint2D_data[i,1], joint2D_data[parent_index,1]))

        plt.text(joint2D_data[i,0], joint2D_data[i,1], H36M_NAMES[i], fontsize=10)
    plt.text(0,0, "Frame : {}".format(frameNum), fontsize=20)

    frameNum = frameNum+1
    plt.pause(.01)
    print(frameNum)



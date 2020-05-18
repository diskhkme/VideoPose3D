import scipy.io
import os
import numpy as np
import struct
import pickle

class ClientDatSimulation():

    def __init__(self, MPI_ROOT_PATH = 'D:/Test_Models/PoseEstim/mpi_inf_3dhp_hkkim_dataset/',
                        CAMERA_NUM = 5,PERSON_NUM = 1,SEQUENCE_NUM = 1,IMG_SOURCE_WIDTH = 2048,IMG_SOURCE_HEIGHT = 2048,
                        IMG_RESIZE_WIDTH = 368,IMG_RESIZE_HEIGHT = 368,HEATMAP_RESIZE_WIDTH = 47,HEATMAP_RESIZE_HEIGHT = 47,
                        HEATMAP_SIGMA = 1,NUM_JOINT = 28, HEADERSIZE = 4):
        self.path = MPI_ROOT_PATH
        self.camera_num = CAMERA_NUM
        self.person_num = PERSON_NUM
        self.img_source_height = IMG_SOURCE_HEIGHT
        self.img_source_width = IMG_SOURCE_WIDTH
        self.header_size = HEADERSIZE

        annotPath = os.path.join(MPI_ROOT_PATH, 'S{0}/Seq{1}/annot.mat'.format(PERSON_NUM, SEQUENCE_NUM))
        annotMat = scipy.io.loadmat(annotPath)

        annot2 = annotMat['annot2']  # 2D 좌표
        annot3 = annotMat['annot3']  # 3D 좌표
        self.annot3_univ = annotMat['univ_annot3']  # Normalized 3D 좌표

        self.joint3D = annot3[CAMERA_NUM][0]  # frame x 84(3*28)
        self.joint2D = annot2[CAMERA_NUM][0]  # frame x 56(2*28)

    def GetDataNext(self, frameIndex):
        joint2DData = self.joint2D[frameIndex,0:48]
        joint2DData = np.concatenate((np.reshape(joint2DData,(24,2)),np.ones((24,1))),axis=1)
        joint2DData = np.reshape(joint2DData,(72,1))
        joint3DData = self.joint3D[frameIndex,0:36] #세 점(controller + HMD 가정)의 위치 정보 3x4 matrx 3개. 현재는 임의의 36개 data 보내도록 가정

        d = struct.pack('i 2I 36f 72f', frameIndex, self.img_source_width, self.img_source_height,
                    *joint3DData, *joint2DData)
        length = len(d)
        msg = bytearray(struct.pack('2i 2I 36f 72f', length, frameIndex, self.img_source_width, self.img_source_height,
                                                *joint3DData, *joint2DData))
        return msg


        # d = [frameIndex, self.img_source_height, self.img_source_width,
        #      joint2DData, joint3DData]
        # msg = pickle.dumps(d)
        # msg = bytes(f"{len(msg):<{self.header_size}}","utf-8") + msg

        # return msg



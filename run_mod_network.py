# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import time

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.utils import deterministic_random

import socket, threading
import struct
import numpy as np
import time
import matplotlib.pyplot as plt

args = parse_args()
print(args)

if args.add3d == True:
    perJointDataDim = 5
else:
    perJointDataDim = 2

model_pos = TemporalModel(17, perJointDataDim, 17,
                          filter_widths=[3,3,3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                          dense=args.dense)


receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()


if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'])


# Evaluate
def evaluate(input_data):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0

        # No batch
        inputs_2d = torch.from_numpy(input_data.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        # Positional model
        predicted_3d_pos = model_pos(inputs_2d)

        return predicted_3d_pos.squeeze(0).cpu().numpy()


# start server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 9999))
s.listen(5)
print(f"Waiting Connection...")
client_socket, addr = s.accept()
print(f"Connection from {addr} has been established!")

# 27개 frame, 17개 관절, 관절당 5개 data (2개 2d joint + 3개 3d position(머리와 손만)
# add3D가 아닌경우, 관절당 2개
databuffer = np.zeros((1,27,17,perJointDataDim))
databufferLastFrame = 0
stackedDataCount = 0

# recv thread
def recv(client_socket, addr):
    global databuffer
    global databufferLastFrame
    global stackedDataCount

    HEADERSIZE = 4
    joint3DData = np.zeros((17,3))
    try:
        full_msg = b""  # is byte
        new_msg = True

        while True:
            msg = client_socket.recv(4)
            if new_msg:
                msglen = struct.unpack('i', msg[:HEADERSIZE])
                msglen = msglen[0]
                new_msg = False

            full_msg += msg

            if len(full_msg) == msglen:
                # d = pickle.loads(full_msg[HEADERSIZE:])
                b = struct.unpack('i 2I 36f 75f', full_msg[HEADERSIZE:])

                currentFrame = b[0]
                Matrices = np.reshape(np.array(b[3:3 + 36]),(3,4,3))

                joint3DData[9, :] = Matrices[0, 3, :] # Index 수정, 여기서는 0이 head라 가정
                joint3DData[13, :] = Matrices[1, 3, :]  # Index 수정, 여기서는 1이 Left hand라 가정
                joint3DData[16, :] = Matrices[2, 3, :]  # Index 수정, 여기서는 2이 Right hand라 가정

                joint2DData = np.reshape(np.array(b[3 + 36:3 + 36 + 75]), (25, 3)) # TODO : joint2DData[0,:]이, (nose x좌표, nose y좌표, nose score) 가 들어있는 벡터여야 함. 순서가 안맞다면 np array를 transpose해야 함


                # "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot"
                # "Spine", "Thorax", "Nose", "Head", "LShoulder", "LElbow", "LWrist",
                # "RShoulder", "RElbow", "RWrist" 순으로 정렬된 데이터가 네트워크에 들어가야 함

                H36M_NAMES = ['']*32
                H36M_NAMES[0] = 'Hip'  # 8
                H36M_NAMES[1] = 'RHip'  # 12
                H36M_NAMES[2] = 'RKnee'  # 13
                H36M_NAMES[3] = 'RFoot'  # 14
                H36M_NAMES[4] = 'LHip'  # 9
                H36M_NAMES[5] = 'LKnee'  # 10
                H36M_NAMES[6] = 'LFoot'  # 11
                H36M_NAMES[7] = 'Spine'  # ??
                H36M_NAMES[8] = 'Thorax'  # 1
                H36M_NAMES[9] = 'Neck/Nose'  # 0
                H36M_NAMES[10] = 'Head'  # ??
                H36M_NAMES[11] = 'LShoulder'  # 2
                H36M_NAMES[12] = 'LElbow'  # 3
                H36M_NAMES[13] = 'LWrist'  # 4
                H36M_NAMES[14] = 'RShoulder'  # 5
                H36M_NAMES[15] = 'RElbow'  # 6
                H36M_NAMES[16] = 'RWrist'  # 7

                set1 = (8,12,13,14,9,10,11,23,1,0,23,2,3,4,5,6,7) # OpenPose 정의를 참고하여 위 순서로 정렬하기 위한 맵핑 1 (23 index인 mapping spine의 경우, 1:1 맵핑이 없어서 임의로 23넣어두고, 아래에서 계산)
                set2 = (8,9,10,11,12,13,14,23,1,0,23,5,6,7,2,3,4) # OpenPose 정의를 참고하여 위 순서로 정렬하기 위한 맵핑 2 (23 index인 mapping spine의 경우, 1:1 맵핑이 없어서 임의로 23넣어두고, 아래에서 계산)

                joint2DData = joint2DData[set1,:2] # TODO : Set 1/Set 2 둘다 테스트 필요

                # Spine과 Head의 경우 애매함. 일단 적당히 계산하여 넣음
                joint2DData[7][0] = (joint2DData[0][0] + joint2DData[8][0]) / 2 # spine x
                joint2DData[7][1] = (joint2DData[0][1] + joint2DData[8][1]) / 2 # spine y
                joint2DData[10][0] = ((joint2DData[9][0] - joint2DData[8][0]) / 2) + (joint2DData[9][0]) # head x
                joint2DData[10][1] = ((joint2DData[9][1] - joint2DData[8][1]) / 2) + (joint2DData[9][1])  # head y

                child_to_parant_dict = {1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 0, 8: 7, 9: 8, 10: 9,
                                        11: 8, 12: 11, 13: 12, 14: 8, 15: 14, 16: 15}


                print(f"frameIndex {currentFrame}, recvd")

                if args.debug_plot == True:
                    plt.cla()
                    plt.axis([0, int(b[1]), 0, int(b[2])])
                    plt.gca().invert_yaxis()
                    # plt.scatter(joint2DData[:,0], joint2DData[:,1])
                    for i in range(17):
                        if i != 0:
                            parent_index = child_to_parant_dict[i]
                            plt.plot((joint2DData[i, 0], joint2DData[parent_index, 0]),
                                     (joint2DData[i, 1], joint2DData[parent_index, 1]))

                        plt.text(joint2DData[i, 0], joint2DData[i, 1], H36M_NAMES[i], fontsize=10)
                    plt.text(0, 0, "Frame : {}".format(currentFrame), fontsize=20)
                    plt.pause(.00001)

                if stackedDataCount < 27:
                    if perJointDataDim == 5:
                        databuffer[0,stackedDataCount,:,:] = np.concatenate((joint2DData,joint3DData),axis=1)
                    else:
                        databuffer[0, stackedDataCount, :, :] = joint2DData
                    stackedDataCount = stackedDataCount+1
                else:
                    databufferLastFrame = currentFrame
                    np.roll(databuffer,1,axis=1)
                    if perJointDataDim == 5:
                        databuffer[0,stackedDataCount-1,:,:] = np.concatenate((joint2DData,joint3DData),axis=1)
                    else:
                        databuffer[0, stackedDataCount-1, :, :] = joint2DData # 버그 수정! -1이 빠져 있었음

                new_msg = True
                full_msg = b""

    except:
        print("except on recv : " , addr)
    finally:
        client_socket.close()

# send thread
def send(client_socket,addr):
    startInference = False
    lastDatabufferLastFrame = 0

    try:
        while True:
            if stackedDataCount >= 27 and startInference == False:
                lastDatabufferLastFrame = databufferLastFrame
                startInference = True

            if lastDatabufferLastFrame != databufferLastFrame and startInference == True:
                databufferCopy = np.copy(databuffer)
                databufferLastFrameCopy = databufferLastFrame

                # Run inference -------------------------------------
                predicted_data = evaluate(databufferCopy)
                predicted_data = predicted_data[0, :, :]
                predicted_data = np.reshape(predicted_data,(51,1))

                d = struct.pack('i 51f', databufferLastFrameCopy, *predicted_data)
                length = len(d)
                msg = bytearray(struct.pack('2i 51f', length, databufferLastFrameCopy, *predicted_data))
                client_socket.send(msg)

                lastDatabufferLastFrame = databufferLastFrameCopy


    except:
        print("except on send : " , addr)
    finally:
        client_socket.close()



# 쓰레드를 이용해서 client 접속 대기를 만들고 다시 accept로 넘어가서 다른 client를 대기한다.
thRecv = threading.Thread(target=recv, args=(client_socket, addr))
thSend = threading.Thread(target=send, args=(client_socket, addr))
thRecv.start()
thSend.start()


while True:
    time.sleep(1)
    pass




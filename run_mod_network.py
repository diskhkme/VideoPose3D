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

args = parse_args()
print(args)

model_pos = TemporalModel(17, 5, 17,
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

databuffer = np.zeros((1,27,17,5)) # 27개 frame, 17개 관절, 관절당 5개 data (2개 2d joint + 3개 3d position(머리와 손만)
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
            msg = client_socket.recv(16)
            if new_msg:
                msglen = struct.unpack('i', msg[:HEADERSIZE])
                msglen = msglen[0]
                new_msg = False

            full_msg += msg

            if len(full_msg) - HEADERSIZE == msglen:
                # d = pickle.loads(full_msg[HEADERSIZE:])
                b = struct.unpack('i 2I 36f 72f', full_msg[HEADERSIZE:])

                currentFrame = b[0]
                Matrices = np.reshape(np.array(b[3:3 + 36]),(3,4,3))

                joint3DData[0, :] = Matrices[0, 3, :] # TODO : Index 수정, 여기서는 0이 head라 가정
                joint3DData[1, :] = Matrices[1, 3, :]  # TODO : Index 수정, 여기서는 1이 hand라 가정
                joint3DData[2, :] = Matrices[2, 3, :]  # TODO : Index 수정, 여기서는 2이 hand라 가정

                joint2DData = np.reshape(np.array(b[3 + 36:3 + 36 + 72]), (24, 3))
                joint2DData = joint2DData[(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),:2] # TODO : Index 수정, 여기서는 0~16이 h36m의 0~16과 대응된다고 가정

                print(f"frameIndex {currentFrame}, recvd") # TODO : Data 가시화?

                if stackedDataCount == 0:
                    databuffer[0,stackedDataCount,:,:] = np.concatenate((joint2DData,joint3DData),axis=1)
                    stackedDataCount = stackedDataCount+1
                elif stackedDataCount < 27:
                    databuffer[0,stackedDataCount,:,:] = np.concatenate((joint2DData,joint3DData),axis=1)
                    stackedDataCount = stackedDataCount+1
                else:
                    databufferLastFrame = currentFrame
                    np.roll(databuffer,1,axis=1)
                    databuffer[0,stackedDataCount-1,:,:] = np.concatenate((joint2DData,joint3DData),axis=1)

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




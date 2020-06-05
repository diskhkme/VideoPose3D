import socket, threading
import pickle
import time
import struct

from ClientDataSimulation import ClientDatSimulation

def recv(client_socket):
    HEADERSIZE = 4
    try:
        # Recv handle
        full_msg = b""  # is byte
        new_msg = True

        while True:
            msg = client_socket.recv(4)
            if new_msg:
                msglen = struct.unpack('i', msg[:HEADERSIZE])
                msglen = msglen[0]
                print(f"new message length: {msglen}")
                new_msg = False

            full_msg += msg

            if len(full_msg) == msglen:
                print("full msg recvd")
                print(full_msg[HEADERSIZE:])

                # d = pickle.loads(full_msg[HEADERSIZE:])
                b = struct.unpack('i 51f', full_msg[HEADERSIZE:])

                frame = b[0]
                inferenced3DData = b[0:]
                print(f"frameIndex {frame}, 3DData {inferenced3DData}")

                new_msg = True
                full_msg = b""


    except:
    # 접속이 끊기면 except가 발생한다.
        print("except on recv")
    finally:
    # 접속이 끊기면 socket 리소스를 닫는다.
        client_socket.close()


def send(client_socket):
    sim = ClientDatSimulation()
    frameIndex = 0

    try:
        while True:
            time.sleep(0.1)
            msg = sim.GetDataNext(frameIndex)
            client_socket.send(msg)
            print(f"send data {frameIndex}")

            frameIndex = frameIndex+1

    except:
    # 접속이 끊기면 except가 발생한다.
        print("except on send")
    finally:
    # 접속이 끊기면 socket 리소스를 닫는다.
        client_socket.close()


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 9999))

frameIndex = 0

thSend = threading.Thread(target=send, args = (s,))
thRecv = threading.Thread(target=recv, args = (s,))

thSend.start()
thRecv.start()

while True:
    time.sleep(1)
    pass

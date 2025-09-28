"""
    Sample camera driver and zmq transmitter
"""

import time
import zmq
import cv2

import osgar.lib.serialize


class ZmqPush:
    def __init__(self, endpoint = "tcp://*:5559"):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 100)  # milliseconds
        self.socket.bind(endpoint)

    def push_msg(self, data):
        raw = osgar.lib.serialize.serialize(data)
        self.socket.send_multipart([bytes("images", 'ascii'), raw])

    def close(self):
        self.socket.close()


def camera(port = 0):
    push = ZmqPush()
    cap = cv2.VideoCapture(port)

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                retval, data = cv2.imencode('*.jpeg', frame)
                if len(data) > 0:
                    push.push_msg(data)
                    time.sleep(1)
    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        push.close()

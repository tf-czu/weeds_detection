"""
    Simulate camera using images stored in a directory
"""

import os
import logging
import time
from threading import Thread

import cv2

g_logger = logging.getLogger(__name__)

class DummyCamera:
    def __init__(self, config, bus):
        self.bus = bus
        self.bus.register("color")
        self.interval = 1.0 / config.get("frequency_hz", 5)
        self.im_dir = config.get("directory")
        if self.im_dir is None:
            g_logger.error(f"No directory was specified!")

        self.last_im_time = None
        self.input_thread = Thread(target=self.run_input, daemon=True)

    def start(self):
        self.input_thread.start()

    def join(self, timeout=None):
        self.input_thread.join(timeout=timeout)

    def load_images(self):
        names = sorted(os.listdir(self.im_dir))
        ret_list = []
        for name in names:
            im_path = os.path.join(self.im_dir, name)
            if os.path.isfile(im_path) and name.endswith(tuple(["tiff", "tif", "png", "jpg", "jpeg"])):
                ret_list.append(im_path)
        return ret_list

    def run_input(self):
        if self.im_dir:
            images_paths = self.load_images()
            n = len(images_paths)
            assert n != 0, images_paths
            self.bus.sleep(1)  # start a little later
            ii = 0
            while self.bus.is_alive():
                img = cv2.imread(images_paths[ii])
                if self.last_im_time:
                    dt = time.time() - self.last_im_time
                    self.bus.sleep(self.interval - dt)
                self.last_im_time = time.time()

                success, encoded_image = cv2.imencode('*.jpeg', img)
                if success:
                    self.bus.publish("color", encoded_image.tobytes())
                ii += 1
                if ii == n:
                    ii = 0

        else:
            self.request_stop()

    def request_stop(self):
        self.bus.shutdown()

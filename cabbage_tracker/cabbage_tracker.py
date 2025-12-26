"""
    Main cabbage detector and tracker
"""
import cv2
import numpy as np
from collections import deque

from osgar.node import Node
from osgar.lib.route import Convertor as GPSConvertor
from cabbage_tracker.detector import Detector

class Cabbage(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('detection')
        self.bus = bus
        self.verbose = False

        model_path = config.get("model")
        self.detector = Detector(model_path)
        self.gps_converter = None
        self.xy_history = deque(maxlen=5)  # keep buffer with desired length.
        self.lonlat_0 = None

    def on_image(self, data):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        # print(image.shape)  # remove
        detections = self.detector.detect(image)
        # print(detections)  # # remove
        # print(list(self.xy_history))  # remove

    def on_nmea_data(self, data):
        if data["quality"] != 0:
            assert data["lon_dir"] == "E"
            assert data["lat_dir"] == "N"
            if self.gps_converter:
                x, y = self.gps_converter.geo2planar((data["lon"], data["lat"]))

            else:
                self.gps_converter = GPSConvertor((data["lon"], data["lat"]))
                self.lonlat_0 = [data["lon"], data["lat"]]
                x = 0
                y = 0
            self.xy_history.append([self.time, [x, y]])


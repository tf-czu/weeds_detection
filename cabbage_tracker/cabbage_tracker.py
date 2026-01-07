"""
    Main cabbage detector and tracker
"""
import math
import cv2
import numpy as np
from collections import deque
import csv
from datetime import datetime
import logging

from osgar.node import Node
from osgar.lib.route import Convertor as GPSConvertor
from cabbage_tracker.detector import Detector

g_logger = logging.getLogger(__name__)


def is_point_in_history(history, x, y, max_dist = 0.05):
    if not history:
        return False

    pts = np.asarray(history, dtype=float)
    p = np.asarray([x, y], dtype=float)

    diff = pts - p
    dist = np.linalg.norm(diff, axis=1)
    return np.any(dist < max_dist)


def rotate_point(x, y, angle_rad):
    # Pre-calculate sine and cosine
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    # Apply rotation matrix formula
    # x' = x*cos(theta) - y*sin(theta)
    # y' = x*sin(theta) + y*cos(theta)
    new_x = x * c - y * s
    new_y = x * s + y * c

    return new_x, new_y


def advance_point(x, y, speed, heading, t0, t1):
    dt = (t1 - t0).total_seconds()
    distance = speed * dt
    dx = distance * math.cos(heading)
    dy = distance * math.sin(heading)

    return x + dx, y + dy


def analyze_trajectory(data):
    """
    Analyzes trajectory data to find heading and speed.
    Args:
        data (list): List of [[datetime, x, y], ...]
    Returns:
        tuple: (heading_rad, speed_mps) or None if checks fail.
    """

    if len(data) != 5:
        return None

    times = [row[0] for row in data]
    x_coords = np.array([row[1][0] for row in data])
    y_coords = np.array([row[1][1] for row in data])

    start_point = np.array([x_coords[0], y_coords[0]])
    end_point = np.array([x_coords[-1], y_coords[-1]])
    dist = np.linalg.norm(end_point - start_point)
    if dist < 0.2:
        return None

    slope, intercept = np.polyfit(x_coords, y_coords, 1)
    heading = math.atan(slope)  # zero in east direction, nord +

    # DIRECTION CORRECTION:
    # atan returns values between -pi/2 and pi/2.
    # We must check the actual movement direction along the X axis.
    if x_coords[-1] < x_coords[0]:
        # Moving left: add 180 degrees (pi)
        heading += math.pi

    # Normalize to range [-pi, pi]
    heading = (heading + math.pi) % (2 * math.pi) - math.pi

    # Calculate average speed
    # Time difference in seconds
    delta_t = (times[-1] - times[0]).total_seconds()
    if delta_t <= 0:
        return None  # Prevent division by zero, in case no move

    speed = dist / delta_t

    return heading, speed


class Cabbage(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('detection', 'cab_pose', 'cab_coordinates')
        self.bus = bus
        self.verbose = False

        model_path = config.get("model")
        self.detector = Detector(model_path)
        self.gps_converter = None
        self.xy_history = deque(maxlen=5)  # keep buffer with desired length.
        self.lonlat_0 = None
        self.frame_size = config.get("frame_size", [2048, 1536])
        fov = config.get("fov", 47.98)
        self.h = 0.45  # Estimated camera distance to surface reduced by plant height
        self.f = self.frame_size[0]/( 2 * math.tan(math.radians( fov/2 )) )  # pixel focus length from camera FOV
        self.cabbage_history = []

        self.csv_file = open(datetime.now().strftime("cab_coordinates_%Y%m%d_%H%M%S.csv"),
                             mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)


    def get_rel_pose(self, cx, cy):
        """
        Keep x on right and y forward
        """
        rel_cx = cx - self.frame_size[0]/2
        rel_cy = self.frame_size[1]/2 - cy
        x = (rel_cx * self.h) / self.f
        y = (rel_cy * self.h) / self.f

        return x, y

    def on_image(self, data):
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
        im_h, im_w, __ = image.shape
        assert (im_h == self.frame_size[1]) and (im_w == self.frame_size[0]), image.shape

        detections = self.detector.detect(image)
        self.publish('detection', detections)

        for (x1, y1, x2, y2), __, score in detections:
            if x1 == 0 or x2 == im_w or y1 == 0 or y2 == im_h:
                continue  # skip detections on borders
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            x, y = self.get_rel_pose(cx, cy)  # relative pose to the camera, x left, y forward.

            machine_heading_speed = analyze_trajectory(list(self.xy_history))
            if machine_heading_speed is not None:
                heading, speed = machine_heading_speed
                t0, (last_x, last_y) = self.xy_history[-1]
                mx, my = advance_point(last_x, last_y, speed, heading, t0, self.time)  # get current machine pose
                # rotate relative object pose according to heading
                rr_x, rr_y = rotate_point(x, y, -math.pi/2 + heading)  # the image is rotated by -pi/2

                # cabbage planer pose
                cab_x = mx + rr_x
                cab_y = my + rr_y

                if not is_point_in_history(self.cabbage_history, cab_x, cab_y):
                    self.cabbage_history.append([cab_x, cab_y])
                    self.publish('cab_pose', [cab_x, cab_y])
                    cab_lon, cab_lat = self.gps_converter.planar2geo([cab_x, cab_y])
                    self.publish('cab_coordinates', [cab_lon, cab_lat])
                    try:
                        self.csv_writer.writerow([cab_lon, cab_lat])
                        self.csv_file.flush()
                    except ValueError:
                        g_logger.warning ('CSV log file is already closed!')
                    if self.verbose:
                        print(cab_x, cab_y)
                        print(cab_lon, cab_lat)


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

    def request_stop(self):
        self.csv_file.close()
        super().request_stop()



import unittest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import math

from cabbage_tracker.cabbage_tracker import Cabbage, analyze_trajectory, is_point_in_history


def make_data(points, t0=None, dt_seconds=1):
    """
    points: list of (x, y) length 5
    """
    if t0 is None:
        t0 = datetime(2026, 1, 1, 12, 0, 0)
    return [
        [t0 + timedelta(seconds=i * dt_seconds), [float(x), float(y)]]
        for i, (x, y) in enumerate(points)
    ]


class TestCabbage(unittest.TestCase):
    def setUp(self):
        bus = MagicMock()
        self.cam = Cabbage(config={"model": "cabbage_tracker/images/best2.pt"}, bus=bus)
        # self.cam.f = 2048/( 2 * math.tan(math.radians( 47.98/2 )) )  # pixel focus length from camera FOV
        # self.cam.h = 0.45

    def assertAngleAlmostEqual(self, a, b, delta=1e-7):
        """
        Compare angles, its useful neer to -pi/pi
        """
        # normalise diff (-pi, pi]
        diff = (a - b + math.pi) % (2 * math.pi) - math.pi
        self.assertLessEqual(abs(diff), delta, msg=f"a={a}, b={b}, diff={diff}")

    def test_center_is_zero(self):
        x, y = self.cam.get_rel_pose(1024, 768)
        print(x, y)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_projection_at_focal_distance(self):
        """
        Move cx in f thus x going to be h
        """
        x, y = self.cam.get_rel_pose(1024 + self.cam.f, 768)
        self.assertAlmostEqual(x, self.cam.h)
        self.assertEqual(y, 0)

    def test_zero_height(self):
        """For zero height all have to be zero"""
        self.cam.h = 0
        x, y = self.cam.get_rel_pose(2000, 1500)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_quadrants(self):
        """ Point on right and in up part, x and y are higher to 0"""
        x, y = self.cam.get_rel_pose(1500, 500)
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)

    def test_heading_east_is_zero(self):
        # slope=0 => heading=0
        data = make_data([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        heading, speed = analyze_trajectory(data)
        self.assertAngleAlmostEqual(heading, 0.0)
        self.assertAlmostEqual(speed, 1)  # 4 meters per 4 seconds

    def test_heading_west_is_minus_pi_due_to_normalization(self):
        # slope=0 => atan=0; correction +pi => pi; normalise => -pi
        data = make_data([(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)])
        heading, _speed = analyze_trajectory(data)
        self.assertAlmostEqual(heading, -math.pi, places=7)

    def test_heading_northeast_pi_over_4(self):
        # y=x, x increase => pi/4
        data = make_data([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        heading, _speed = analyze_trajectory(data)
        self.assertAngleAlmostEqual(heading, math.pi / 4)

    def test_heading_southeast_minus_pi_over_4(self):
        # y = -x + 4, x increase => -pi/4
        data = make_data([(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
        heading, _speed = analyze_trajectory(data)
        self.assertAngleAlmostEqual(heading, -math.pi / 4)

    def test_heading_changes_by_pi_when_reversing_x_direction_for_same_slope(self):
        # The same slope (slope=1), but different direction.
        data_ne = make_data([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        data_sw = make_data([(4, 4), (3, 3), (2, 2), (1, 1), (0, 0)])

        heading_ne, _ = analyze_trajectory(data_ne)  # ~ +pi/4
        heading_sw, _ = analyze_trajectory(data_sw)  # ~ -3pi/4

        self.assertAngleAlmostEqual(heading_ne, math.pi / 4)
        self.assertAngleAlmostEqual(heading_sw, -3 * math.pi / 4)

    def test_returns_true_when_point_within_threshold_exists(self):
        history = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
        # Distance to [1.0, 1.0] is 0.03 < 0.05
        self.assertTrue(is_point_in_history(history, 1.03, 1.0, max_dist=0.05))

    def test_returns_false_when_all_points_outside_threshold(self):
        history = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
        # Closest is [1.0, 1.0], distance sqrt(0.05^2 + 0^2) == 0.05 -> not strictly less
        self.assertFalse(is_point_in_history(history, 1.05, 1.0, max_dist=0.05))


if __name__ == '__main__':
    unittest.main()

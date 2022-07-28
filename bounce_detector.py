from collections import deque
from typing import Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utils import utilities
from utils.court import Court
from utils.rect import Rect


class BounceDetector:
    """
    Implements ball bounce detection and bounce visualisation based on ball contour path tracking.
    """

    def __init__(self, src: list, dst: list):

        self.src, self.__court_lower_boundary_L, self.__court_lower_boundary_R = self.__reorder_src_coords(src)
        self.dst = dst
        self.__remap_dst_coords(self.dst)

        # Modify service box bottom coordinates for more accurate mapping
        # An intersection is taken with the line defined by the lower boundary of the court and
        # the service box vertical lines.
        # The lower coordinates of the service box corners are then changed to lower coordinates of the court.

        # Thus this allows to directly map the court via homography and create a 1-to-1 mapping between a court
        # image and the bounce location in the homography image.
        self.src[0] = utilities.get_intersect(self.src[0], self.src[1], self.__court_lower_boundary_L[:2],
                                              self.__court_lower_boundary_R[:2])
        self.src[-1] = utilities.get_intersect(self.src[2], self.src[3], self.__court_lower_boundary_L[:2],
                                               self.__court_lower_boundary_R[:2])
        # endregion

        self.__homography_matrix, _ = cv.findHomography(self.src, self.dst, cv.RANSAC, 5.0)
        self.__contour_path_history = deque(maxlen=5)
        # Fill with initial dummy values
        for i in range(self.__contour_path_history.maxlen):
            self.__contour_path_history.append([0, 0])

        # BOUNCE_COOLDOWN determines the number of frames that must pass between subsequent frames
        # before another bounce can be registered.
        # The given constant has been hand-picked to be reasonable for 60fps video.
        self.__BOUNCE_COOLDOWN = 80  # Number of frames
        # Keeps track of cooldown progress
        self.__bounce_cooldown_counter = 0

        # Flag for __plot_ball_method initialization
        self.__initialized_plotting = False

    def bounced(self) -> bool:
        """
        :return: True, if ball bounced, False otherwise.
        """
        """
        A ball bounce is recognized by:
        1) Cooldown for ball bounce-recognition expiring and 
        2) Noticing trend of ball height-coordinate increasing switching to decreasing.
        
        The cooldown is implemented to only recognize ball bounces that first contact the floor.
        If this was not implemented, then secondary bounces (such as wall -> glass) would also be mistakenly
        be recognized as bounces.
        """
        _, current_projected_y = self.__contour_path_history[-1]
        _, previous_projected_y = self.__contour_path_history[0]

        if self.__bounce_cooldown_counter < 0:  # Only detect bounces once the cooldown has refreshed
            # Spotting the peak of the bounce
            for x_proj, y_proj in self.__contour_path_history:
                if not utilities.is_within_window_height(y_proj):
                    return False

            if self.__contour_path_history[0][1] <= self.__contour_path_history[1][1] < self.__contour_path_history[2][
                1] \
                    and self.__contour_path_history[2][1] > self.__contour_path_history[3][1] >= \
                    self.__contour_path_history[4][1]:
                self.__bounce_cooldown_counter = self.__BOUNCE_COOLDOWN
                return True
        return False

    def update_contour_data(self, contour: Rect) -> None:
        """
        Add data to the detector for bounce detection.
        :param contour: Ball contour
        """
        contour_projection_point = self.__project_point(
            (contour.x + contour.width / 2, contour.y + contour.height / 2, 1))
        self.__bounce_cooldown_counter -= 1
        self.__contour_path_history.append(contour_projection_point)

        # For real-time plotting uncomment:
        # self.__plot_ball_path()

    def get_last_bounce_location(self) -> (int, int):
        """
        WARNING: This method returns valid data only if bounced() returns true.
        This method retrieves the last known bounce location of the ball.
        :return: 2D ball coordinates
        """
        # We pick NOT the current contour, but the middle known contour, because
        # the current contour already signals the next positions from the bounce whereas
        # the middle contour actually marks the position of the bounce.
        return int(self.__contour_path_history[2][0]), int(self.__contour_path_history[2][1])

    def __project_point(self, point):
        """
        Projects a single point from src coordinates to dst coordinates based on the homography matrix
        calculated during class initialization.
        :param point: Point of the form: [x: float, y: float, 1]
        :return: Projection of point from src coordinates to dst coordinates.
        """
        return self.__homography_matrix[0: 2] @ point / (self.__homography_matrix[2:] @ point)

    def show_projection(self, frame: np.ndarray) -> None:
        """
        Displays the top-down view by the calculated homography.
        :param frame: Frame to be shown as projected by the homography.
        """
        cv.imshow("Projection view",
                  cv.warpPerspective(frame, self.__homography_matrix, (frame.shape[1], frame.shape[0])))

    def __plot_ball_path(self) -> None:
        """
        Utility function for visualizing ball bounce characteristics.
        """
        if not self.__initialized_plotting:
            self.frame_count = [0]
            self.y_coords = [0]
            self.fig = plt.figure()
            self.fig.set_figheight(5)
            self.fig.set_figwidth(7)
            self.ax = self.fig.gca()
            plt.show(block=False)
            self.__initialized_plotting = True

        self.y_coords.append(self.__contour_path_history[-1][1])
        self.frame_count.append(self.frame_count[-1] + 1)

        self.ax.clear()
        self.ax.plot(self.frame_count, self.y_coords, label="Ball contour projection y-coordinate during frame x")

        plt.title('Ball bounce characterisation')
        plt.xlabel('Video frame')
        plt.ylabel('Contour projection y-height')
        plt.legend()
        plt.axis([50, None, 0, 740])
        self.fig.canvas.draw()

    def __reorder_src_coords(self, src_coords: list) -> Tuple:
        """
        :param src_coords: 6 source coordinates [4 box coordinates, 2 boundary coordinates]
        :return: Reordered coordinates starting from left lower service box corner going clockwise and court boundaries'
        coordinates left and right.
        """

        box_coords, court_boundary_coords = src_coords
        y_key = lambda x: x[1]
        x_key = lambda x: x[0]

        # Enforce an ordering starting from left lower service box going clockwise.
        box_sorted_by_x = sorted(box_coords, key=x_key)
        # y-coordinate grows downwards, lower has higher y than upper
        left_lower = max(box_sorted_by_x[0], box_sorted_by_x[1], key=y_key)
        left_upper = min(box_sorted_by_x[0], box_sorted_by_x[1], key=y_key)
        right_upper = min(box_sorted_by_x[2], box_sorted_by_x[3], key=y_key)
        right_lower = max(box_sorted_by_x[2], box_sorted_by_x[3], key=y_key)

        boundary_sorted_x = sorted(court_boundary_coords, key=lambda x: x[0])
        boundary_L = min(*boundary_sorted_x, key=x_key) + (1,)
        boundary_R = max(*boundary_sorted_x, key=x_key) + (1,)

        return np.array([left_lower, left_upper, right_upper, right_lower]), boundary_L, boundary_R

    def __remap_dst_coords(self, dst) -> None:
        """
        Remaps dst coords from service box corners to service box corners and lower court boundary
        """
        dst[0] = (dst[0][0], Court.side_wall_len)
        self.dst[1] = (dst[1][0], Court.short_line_from_front_wall)
        self.dst[2] = (dst[2][0], Court.short_line_from_front_wall)
        dst[3] = (dst[3][0], Court.side_wall_len)
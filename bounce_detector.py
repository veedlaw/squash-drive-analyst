import numpy as np
import cv2 as cv
from collections import namedtuple, deque
from matplotlib import pyplot as plt
# from utilities import get_intersect, is_within_window_height, FRAME_HEIGHT, FRAME_WIDTH
import utilities

from rect import Rect


class BounceDetector:
    """
    Implements ball bounce detection and bounce visualisation based on ball contour path tracking.
    """

    def __init__(self, src, dst):
        self.Point = namedtuple('Point', 'x, y')

        # region
        # Hardcoded for development speed purposes
        self.src = np.array([
            (170, 494),
            (203, 460),
            (340, 507),
            (340, 466)
        ])

        # self.dst = np.array([
        #     (270, 462),
        #     (270, 357),
        #     (360, 462),
        #     (360, 357)
        # ])

        # self.src = np.array([
        #     # (170, 494),
        #     (203, 460),
        #     (160, 494),
        #     # (348, 502),
        #     (349, 634),
        #     (340, 465)
        # ])
        self.dst = np.array([
            (270, 640),
            (270, 357),
            (360, 640),
            (360, 357)
        ])

        # Hardcoded for development speed purposes
        # Real video coordinates
        self.__court_lower_boundary_L = (3, 603, 1)
        self.__court_lower_boundary_R = (354, 636, 1)

        print(self.__court_lower_boundary_L[:2])
        print(self.__court_lower_boundary_R[:2])
        print(self.src[0])
        print(self.src[1])
        # Modify source coordinates for more accurate mapping
        self.src[0] = utilities.get_intersect(self.src[0], self.src[1], self.__court_lower_boundary_L[:2],
                                              self.__court_lower_boundary_R[:2])
        self.src[2] = utilities.get_intersect(self.src[2], self.src[3], self.__court_lower_boundary_L[:2],
                                              self.__court_lower_boundary_R[:2])

        self.__homography_matrix, _ = cv.findHomography(self.src, self.dst, cv.RANSAC, 5.0)
        self.__contour_path_history = deque(maxlen=2)
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
        # endregion

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
            if utilities.is_within_window_height(current_projected_y) and \
                    utilities.is_within_window_height(previous_projected_y) and \
                    current_projected_y < previous_projected_y:
                # Since the bounce has been registered, we reset the cooldown.
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
        # We pick NOT the current contour, but the previous known contour, because
        # the current contour already signals the next position from the bounce whereas
        # the previous contour actually marks the position of the bounce.
        return int(self.__contour_path_history[0][0]), int(self.__contour_path_history[0][1])

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

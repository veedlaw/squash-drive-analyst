import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utils.rect import Rect

FRAME_WIDTH = 360
FRAME_HEIGHT = 640


def show_histogram(img):
    """Debugging function that shows plotted histogram of two images."""
    hist_full = cv.calcHist([img], [0], None, [255], [0, 255])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.plot(hist_full)
    plt.xlim([0, 255])
    plt.show()



def draw_rect(frame: np.ndarray, rect: Rect, color: (int, int, int), line_width=2) -> None:
    cv.rectangle(frame, (int(rect.x), int(rect.y)), (int(rect.x) + int(rect.width), int(rect.y) + int(rect.height)),
                 color, line_width)


def is_within(rect: Rect, x: float, y: float) -> bool:
    """
    :param rect: Rectangle
    :param x: X-coordinate
    :param y: Y-coordinate
    :return: True if (x, y) lies within the boundaries of the rect. False otherwise.
    """
    return (rect.x <= x <= rect.x + rect.width) and (rect.y <= y <= rect.y + rect.height) or \
           ((rect.x >= x >= rect.x + rect.width) and (rect.y <= y <= rect.y + rect.height))


def get_intersect(p1: (float, float), p2: (float, float), q1: (float, float), q2: (float, float)) -> (float, float):
    """
    :param p1: (x, y) first point on the first line
    :param p2: (x, y) second point on the first line
    :param q1: (x, y) first point on the second line
    :param q2: (x, y) second point on the second line
    :return: Point of intersection of the lines passing through p1, p2 and q1, q2
    """
    """
    Based on:
        https://medium.com/@unifyai/part-i-projective-geometry-in-2d-b1ca26d5fa2a
        https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    """
    # Convert all Cartesian points to homogeneous coordinates
    h = np.hstack((np.vstack([p1, p2, q1, q2]), np.ones((4, 1))))
    # get the first line
    l1 = np.cross(h[0], h[1])
    # get the second line
    l2 = np.cross(h[2], h[3])
    # point of intersection
    x, y, z = np.cross(l1, l2)
    # lines are parallel
    if z == 0:
        return float('inf'), float('inf')
    return x / z, y / z


def is_within_window_height(y_height: float) -> bool:
    """
    :param y_height: Coordinate to check
    :return: True if given y lies in the frame boundary, false otherwise
    """
    return 0 <= y_height <= FRAME_HEIGHT

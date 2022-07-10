import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils.rect import Rect

FRAME_WIDTH = 360
FRAME_HEIGHT = 640


def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=4, pxstep=90, pystep=128):
    """
    Draws a grid on an image
    :param img: Image for the lines to be drawn on.
    :param line_color: Color of the line
    :param thickness: Thickness of the line
    :param type_: Type of line
    :param pxstep: Every pxstep pixels a line will be drawn
    :param pystep: Every pystep pixels a line will be drawn
    """
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep
    return img


def show_histogram(img, processed_img):
    """Debugging function that shows plotted histogram of two images."""
    hist_full = cv.calcHist([img], [0], None, [255], [0, 255])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(processed_img, 'gray')
    plt.subplot(223), plt.plot(hist_full)
    plt.xlim([0, 255])
    plt.show()


def blend(list_images):  # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output


def draw_rect(frame: np.ndarray, rect: Rect, color: (int, int, int), line_width=2) -> None:
    cv.rectangle(frame, (int(rect.x), int(rect.y)), (int(rect.x) + int(rect.width), int(rect.y) + int(rect.height)),
                 color, line_width)


def is_within(rect1: Rect, rect2: Rect) -> bool:
    # print(f"{rect1.x} + {rect1.width} > {rect2.x} or {rect2.x} + {rect2.width} > {rect1.x}")
    return (rect1.x < (rect2.x + rect2.width) and rect2.x < (rect1.x + rect1.width)) \
           and (rect1.y < (rect2.y + rect2.height) and rect2.y < (rect1.y + rect1.height))


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

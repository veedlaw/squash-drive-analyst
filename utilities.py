import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from rect import Rect

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


def draw_rect(frame: np.ndarray, rect: Rect, color: (int, int, int)) -> None:
    cv.rectangle(frame, (int(rect.x), int(rect.y)), (int(rect.x) + int(rect.width), int(rect.y) + int(rect.height)),
                 color, 2)


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
    :return: Point of intersection ofthe lines passing through p1, p2 and q1, q2
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


def draw_court() -> np.ndarray:
    """
    Draws a squash court using OpenCV drawing functions.
    :return: Drawing of the court
    """
    """
           ...                    side wall
                   5.44m --------> |
                                   |
                                   |
           |----1.525m---x---1.6m----x \\  <------------ Front of service box, Short line 
           |           |           |    } 1.6m
           |           |           | //
           |           M-----------M   <-------------- Back of service box
           |                 |     |\
           |           service box |  \
           |                       |    \\-  
           |     back of court     |    //--  2.61m
           |            |          |  /
           |-----------------------|/
           L---- Half-court line

           Total y-length = 9.75m (5.44m + 1.6m + 2.61m)
    """
    court_img = np.empty((640, 360, 3), dtype=np.uint8)
    COLOR_COURT = (181, 218, 240)
    COLOR_RED = (0, 0, 255)
    LINE_WIDTH = 3
    court_img[:] = COLOR_COURT

    # Real side wall length is 9.75m or in a 1-to-1 conversion 975px
    # similarly the read front wall length is 6.40m or 640px
    # However we want a 1-to-1 mapping with our video resolution
    # so we perform the conversions:
    side_wall_len = FRAME_HEIGHT
    front_wall_len = FRAME_WIDTH

    hConv = side_wall_len / 975
    wConv = front_wall_len / 640

    short_line_from_front_wall = int(544 * hConv)
    service_box_len = 160

    service_box_front_outer_L = (0, short_line_from_front_wall)
    service_box_back_inner_L = (int(service_box_len * wConv), short_line_from_front_wall + int(service_box_len * hConv))
    service_box_front_outer_R = (front_wall_len, short_line_from_front_wall)
    service_box_back_inner_R = (front_wall_len - int(service_box_len * wConv),
                                short_line_from_front_wall + int(service_box_len * hConv))

    # Draw the "Short line"
    cv.line(court_img, service_box_front_outer_L, service_box_front_outer_R, COLOR_RED, LINE_WIDTH)

    # Draw the "Half Court line"
    half_court_line_mid_court = (int((service_box_len + 152.5) * wConv), short_line_from_front_wall)
    half_court_line_end_court = (int((service_box_len + 152.5) * wConv), side_wall_len)
    cv.line(court_img, half_court_line_mid_court, half_court_line_end_court, COLOR_RED, LINE_WIDTH)

    # Draw the Left side service box
    cv.rectangle(court_img, service_box_front_outer_L, service_box_back_inner_L, COLOR_RED, LINE_WIDTH)

    # Draw the Right side service box
    cv.rectangle(court_img, service_box_front_outer_R, service_box_back_inner_R, COLOR_RED, LINE_WIDTH)

    return court_img


def draw_ball_projection(court: np.ndarray, x: int, y: int) -> None:
    """
    Draws a circle on court at location (x,y)
    :param court: Image to be drawn on
    :param x: Center-coordinate x
    :param y: Center-coordinate y
    """

    chunk_size = 11
    circle_radius = 6

    # Make sure x, y are within bounds
    if x >= FRAME_WIDTH:
        x = FRAME_WIDTH - chunk_size
    elif x <= chunk_size:
        x = chunk_size

    if y >= FRAME_HEIGHT:
        y = FRAME_HEIGHT - chunk_size
    elif y <= chunk_size:
        y = chunk_size

    # Select a chunk of the court
    court_chunk = court[y - chunk_size // 2: y + chunk_size // 2 + 1, x - chunk_size // 2: x + chunk_size // 2 + 1]
    court_chunk_copy = np.copy(court_chunk)

    # Draw the ball as a circle on the copied chunk
    cv.circle(court_chunk_copy, center=(chunk_size // 2, chunk_size // 2),
              radius=circle_radius, color=(255, 0, 0), thickness=-1)
    # # Introduce transparency
    chunk_with_circle = cv.addWeighted(court_chunk, 0.7, court_chunk_copy, 0.3, 1.0)
    # Store the chunk back into the original court image
    court[y - chunk_size // 2: y + chunk_size // 2 + 1,
    x - chunk_size // 2: x + chunk_size // 2 + 1] = chunk_with_circle


def is_within_window_height(y_height: float) -> bool:
    """
    :param y_height: Coordinate to check
    :return: True if given y lies in the frame boundary, false otherwise
    """
    return 0 <= y_height <= FRAME_HEIGHT

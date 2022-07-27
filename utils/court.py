from typing import List

import numpy as np
import cv2 as cv
from utils import utilities
from utils.rect import Rect


class Court:
    """
    Implements squash court specific drawing methods.
    """

    """
    Squash court dimensions as specified by WSF (World Squash Federation)
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

    # region court constants
    # Real side wall length is 9.75m or in a 1-to-1 conversion 975px
    # similarly the read front wall length is 6.40m or 640px
    # However we want a 1-to-1 mapping with our video resolution
    # so we perform the conversions:
    side_wall_len = utilities.FRAME_HEIGHT
    front_wall_len = utilities.FRAME_WIDTH

    COLOR_COURT = (181, 218, 240)

    hConv = side_wall_len / 975
    wConv = front_wall_len / 640

    short_line_from_front_wall = int(544 * hConv)
    service_box_len = 160

    service_box_front_outer_L = (0, short_line_from_front_wall)
    service_box_back_inner_L = (int(service_box_len * wConv), short_line_from_front_wall + int(service_box_len * hConv))
    service_box_front_outer_R = (front_wall_len, short_line_from_front_wall)
    service_box_back_inner_R = (front_wall_len - int(service_box_len * wConv),
                                short_line_from_front_wall + int(service_box_len * hConv))

    half_court_line_mid_court = (int(2 * service_box_len * wConv), short_line_from_front_wall)
    half_court_line_end_court = (int(2 * service_box_len * wConv), side_wall_len)

    # endregion court constants

    @staticmethod
    def get_court_drawing() -> np.ndarray:
        """
        Draws a squash court using OpenCV drawing functions.
        :return: Drawing of the court
        """

        court_img = np.empty((utilities.FRAME_HEIGHT, utilities.FRAME_WIDTH, 3), dtype=np.uint8)
        court_img[:] = Court.COLOR_COURT
        COLOR_LINE = (0, 0, 255)
        LINE_WIDTH = 3

        # Draw the "Short line"
        cv.line(court_img, Court.service_box_front_outer_L, Court.service_box_front_outer_R, COLOR_LINE, LINE_WIDTH)

        # Draw the "Half Court line"
        cv.line(court_img, Court.half_court_line_mid_court, Court.half_court_line_end_court, COLOR_LINE, LINE_WIDTH)

        # Draw the Left side service box
        cv.rectangle(court_img, Court.service_box_front_outer_L, Court.service_box_back_inner_L, COLOR_LINE, LINE_WIDTH)

        # Draw the Right side service box
        cv.rectangle(court_img, Court.service_box_front_outer_R, Court.service_box_back_inner_R, COLOR_LINE, LINE_WIDTH)

        return court_img

    @staticmethod
    def draw_targets_grid(court_img: np.ndarray, target_rects: List[Rect]) -> None:
        """
        Draws all target_rects onto court_img.
        :param court_img: Court drawing
        :param target_rects: List of rects to be drawn
        """
        for rect in target_rects:
            utilities.draw_rect(court_img, rect, color=(0, 0, 0), line_width=1)

    @staticmethod
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
        if x >= utilities.FRAME_WIDTH:
            x = utilities.FRAME_WIDTH - chunk_size
        elif x <= chunk_size:
            x = chunk_size

        if y >= utilities.FRAME_HEIGHT:
            y = utilities.FRAME_HEIGHT - chunk_size
        elif y <= chunk_size:
            y = chunk_size

        # Select a chunk of the court
        y_chunk_lower = y - chunk_size // 2
        y_chunk_upper = y + chunk_size // 2 + 1
        x_chunk_lower = x - chunk_size // 2
        x_chunk_upper = x + chunk_size // 2 + 1

        court_chunk = court[y_chunk_lower: y_chunk_upper, x_chunk_lower: x_chunk_upper]
        court_chunk_copy = np.copy(court_chunk)

        # Draw the ball as a circle on the copied chunk
        cv.circle(court_chunk_copy, center=(chunk_size // 2, chunk_size // 2),
                  radius=circle_radius, color=(255, 0, 0), thickness=-1)
        # # Introduce transparency
        chunk_with_circle = cv.addWeighted(court_chunk, 0.7, court_chunk_copy, 0.3, 1.0)
        # Store the chunk back into the original court image
        court[y_chunk_lower: y_chunk_upper, x_chunk_lower: x_chunk_upper] = chunk_with_circle

    @staticmethod
    def create_target_rects() -> List[Rect]:
        """
        TODO
        :return:
        """

        dir = 1
        zone1_R = Rect(x=int(Court.half_court_line_mid_court[0]),
                       y=Court.short_line_from_front_wall,
                       width=dir * int(Court.service_box_len * Court.wConv),
                       height=int(Court.service_box_len * Court.hConv))
        zone2_R = Rect(zone1_R.x, zone1_R.y + zone1_R.height, zone1_R.width, int((261 / 2) * Court.hConv))
        zone3_R = Rect(zone1_R.x, zone2_R.y + zone2_R.height, zone2_R.width, int((261 / 2) * Court.hConv) + 8)
        rects = [zone1_R, zone2_R, zone3_R]
        widths = [dir * int((Court.service_box_len - 30) * Court.wConv), dir * int(40 * Court.wConv)]
        for i in range(2):
            parent_rect = rects[-1]
            for j in range(3):
                rects.append(Rect(parent_rect.x + parent_rect.width, rects[j].y, widths[i], rects[j].height))

        return rects

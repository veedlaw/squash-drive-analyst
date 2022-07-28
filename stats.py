from typing import List

import cv2 as cv
import numpy as np

from utils import utilities
from utils.rect import Rect


class AccuracyStatistics:

    def __init__(self, target_rects):
        # Create a "bucket" for non-target shots
        self.non_target_rect = Rect(0, 0, 0, 0)
        target_rects.insert(0, self.non_target_rect)

        # Maps each target rect to a list of ball bounces
        self.__target_rects = {key: [] for key in target_rects}
        self.__total_shots = 0

        # Marks the naming target_rects
        self.__box_index_start = ord('A')

    def record_bounce(self, x, y) -> None:
        """
        Records a ball bounce location into a target box
        :param x: Ball bounce x coordinate
        :param y: Ball bounce y coordinate
        """

        # Find which target box the bounce landed in and record the bounce
        for target_rect in self.__target_rects.keys():
            if utilities.is_within(target_rect, x, y):
                self.__target_rects[target_rect].append((x, y))
                break
        else:
            self.__target_rects[self.non_target_rect].append((x, y))

        self.__total_shots += 1

    def get_target_rects(self) -> List[Rect]:
        """
        :return: List of tracked target rectangles.
        """
        return [target_rect for target_rect in self.__target_rects.keys()]

    def get_box_to_num_shots(self) -> dict:
        """
        :return: A mapping from each target rect to number of shots bounced in given target rect.
        """

        return {target_rect: len(shots) for target_rect, shots in self.__target_rects.items()}

    def get_result_str_boxwise(self) -> str:
        """
        Constructs a string showing the results of the analysis.
        :return: Formatted string of results.
        """
        result = []
        bounce_data = self.get_box_to_num_shots()
        total_bounces = self.__total_shots

        num_bounces_other = len(self.__target_rects[self.non_target_rect])
        result.append(f"OTHER: \t{num_bounces_other / total_bounces * 100:.1f}% \t{num_bounces_other}/{total_bounces}\n\n")

        count = self.__box_index_start
        for box in bounce_data.keys():
            if box == self.non_target_rect:
                continue
            num_bounces = len(self.__target_rects[box])
            result.append(f"{chr(count)}:\t {(num_bounces / total_bounces) * 100:.1f}%  \t{num_bounces}/{total_bounces}\n\n")
            count += 1

        return ''.join(result)

    def draw_box_markings(self, court_img: np.ndarray) -> None:
        """
        Draws letters on court image corresponding to the letters used in get_result_str_boxwise()
        :param court_img: Image to draw on
        """
        count = self.__box_index_start
        for box in self.get_box_to_num_shots().keys():
            if box == self.non_target_rect:
                continue
            text_x_coord = box.x if box.width > 0 else box.x - 15
            text_y_coord = box.y + int(box.height / 5)

            cv.putText(court_img, str(chr(count)), (text_x_coord, text_y_coord),
                       cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1, cv.LINE_AA)

            count += 1

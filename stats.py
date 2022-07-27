from typing import List

from utils import utilities
from utils.rect import Rect
import cv2 as cv


class AccuracyStatistics:

    def __init__(self, target_rects):
        # Create a "bucket" for non-target shots
        self.non_target_rect = Rect(0, 0, 0, 0)
        target_rects.append(self.non_target_rect)

        # Maps each target rect to a list of ball bounces
        self.__target_rects = {key: [] for key in target_rects}
        print(self.__target_rects)
        self.__total_shots = 0

    def record_bounce(self, x, y) -> None:
        """
        :param x:
        :param y:
        :return:
        """
        i = 0
        for target_rect in self.__target_rects.keys():
            i += 1
            if utilities.is_within(target_rect, x, y):
                print(f"bounced in target block: {i}")
                self.__target_rects[target_rect].append((x, y))
                break
        else:
            self.__target_rects[self.non_target_rect].append((x, y))
        self.__total_shots += 1

    def generate_output(self):
        for i, target_rect in enumerate(self.__target_rects.keys()):
            print(f"Target block {i}: {len(self.__target_rects[target_rect])} shots landed")

        pass

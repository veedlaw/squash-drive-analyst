from typing import List

from utils import utilities
from utils.court import Court
from utils.rect import Rect
import cv2 as cv


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

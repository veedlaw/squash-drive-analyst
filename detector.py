from __future__ import print_function

import cv2 as cv
import random as rng


class Detector:
    rng.seed(12345)

    def classify(self, frame):
        """
        :param frame: a preprocessed image
        :return:
        """


        player_candidates = []
        ball_candidates = []

        # RETR external doesn't store any contours within contours
        # finds contours within the segmented image
        cnt, hierarchy = cv.findContours(cv.Canny(frame, 0, 1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        areas = []
        if len(cnt) > 1:
            for c in cnt:
                areas.append(cv.contourArea(c))


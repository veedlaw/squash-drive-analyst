from __future__ import print_function
from operator import itemgetter

import cv2
import cv2 as cv
import numpy as np


class Detector:

    def classify(self, frame):
        """
        :param frame: A preprocessed image
        :return:
        """

        player_candidates = []
        ball_candidates = []

        cleaned_contours = self.__join_contours(frame)
        # min_contour = min(cleaned_contours, key=rect_area)
        # print(f'min contour: {min_contour}')

        return cleaned_contours

    def __join_contours(self, frame: np.ndarray) -> list:
        """"
        :param frame: A preprocessed frame

        The method will receive a preprocessed image, which is likely to contain many contours due to the nature of
        the image segmentation process.
        Given the number of contours, it will be hard to identify which one is likely to be the ball and which are
        noise from the player movement.

        A heuristic to determine the ball contour is that the ball contour is mostly farther away from the player's body
        than the noisy segmentation of the player. Therefore, we will join all contours that are close to each other
        into one box.

        The result should ideally be one large and one small bounding box, that is, the player- and ball candidate
        respectively.
        """

        # Obtain all contours from the image
        contours, _ = cv.findContours(cv.Canny(frame, 0, 1), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            bounding_boxes.append(cv.boundingRect(contour))

        # Sort the bounding boxes according to their x-coordinate in increasing order
        bounding_boxes.sort(key=itemgetter(0))

        bounding_boxes = self.__join_nearby_bounding_boxes(bounding_boxes)

        return bounding_boxes

    def __join_nearby_bounding_boxes(self, bounding_boxes) -> list:
        """
        :param bounding_boxes: Sorted list of bounding_boxes(rectangles) (4-tuple (top-left x, top-left y, width, height))
        :return: List of rectangles

        Many thanks to user HansHirse on StackOverflow.
        """

        join_distance = 5
        processed = [False] * len(bounding_boxes)
        new_bounds = []

        for i, rect1 in enumerate(bounding_boxes):
            if not processed[i]:

                processed[i] = True
                current_x_min, current_y_min, current_x_max, current_y_max = self.__get_rectangle_contours(rect1)

                for j, rect2 in enumerate(bounding_boxes[(i + 1):], start=(i + 1)):

                    candxMin, candyMin, candxMax, candyMax = self.__get_rectangle_contours(rect2)

                    if candxMin <= current_x_max + join_distance:
                        processed[j] = True

                        # Reset coordinates of current rect
                        current_x_max = candxMax
                        # currxMax = max(currxMax, candxMax) - min(currxMin, candxMin)
                        current_y_min = min(current_y_min, candyMin)
                        current_y_max = max(current_y_max, candyMax)
                    else:
                        break
                new_bounds.append([current_x_min, current_y_min,
                                   current_x_max - current_x_min, current_y_max - current_y_min])

        return new_bounds

    def __get_rectangle_contours(self, rectangle: list) -> list:
        """
        Takes a rectangle and returns the coordinates of the top-left and
        bottom-right corner.

        :param rectangle: 4-tuple (top-left x, top-left y, width, height)
        :return: [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        """
        x_min, y_min, width, height = rectangle
        x_max = x_min + width
        y_max = y_min + height

        return [x_min, y_min, x_max, y_max]


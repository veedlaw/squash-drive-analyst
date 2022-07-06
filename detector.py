import sys
from operator import itemgetter
from collections import deque

from typing import List

from rect import Rect

import cv2 as cv
import numpy as np


class Detector:
    """
    Implements selection of the most probable ball contour from a list of contours.
    """

    def __init__(self):
        self.__candidate_history = deque(maxlen=9)  # deque(list[Rect], list[Rect], ...)

        # Dummy entries for initial start-up of the detector.
        dummy_candidate = Rect(0, 0, 0, 0)
        # Means that during frame 1, we had a single ball candidate: 'dummy candidate'
        self.__candidate_history.append([dummy_candidate])  # dummy entry
        # Similarly, means that during frame 2, we also had single ball candidate: 'dummy candidate'
        self.__candidate_history.append([dummy_candidate])  # dummy entry

        """ Mapping: goal: Rect -> (total_distance_required: float, from_rect: Rect, from_rect_layer_number: int) 
        total_distance_required gives the distance from layer 1 to reach current goal Rect. """
        self.__best_paths = dict()

    def select_most_probable_candidate(self, frame: np.ndarray, prediction: Rect) -> Rect:
        """
        Selects the contour from the frame that most likely appears to be a ball candidate.

        :param frame: Binarized video frame containing contours.
        :param prediction: Predicted contour of the ball in the frame.
        :returns: Contour in image corresponding to ball
        """

        # Clean the contours and store suitable candidates as ball candidates.
        self.__update_ball_candidates(frame, prediction)

        # Obtain the best ball candidate by searching for most continuous path
        # through the previous and up-to-current ball candidates.
        best_candidate = self.__find_shortest_path_candidate()

        # best_candidate is None in case of no candidates at all -> prediction is
        # selected as the most probable ball candidate.
        if best_candidate is None:
            return prediction

        return best_candidate

    def __update_ball_candidates(self, frame: np.ndarray, prediction: Rect) -> None:
        """
        Process contours in the frame and update list of ball candidates from the contours.
        :param prediction: Predicted Rect ball candidate
        :param frame: Binarized video frame containing contours.
        """

        # Reduce noise by joining together nearby contours
        cleaned_contours = self.__join_contours(frame)

        # Sort the contours in ascending order based on contour area
        # (Ideally the largest contour is the player and the smallest contour is the ball)
        cleaned_contours.sort(key=lambda rect: rect.area())

        # Throw away the biggest contour (most likely to be the player)
        ball_candidates = cleaned_contours[:(len(cleaned_contours) - 1)]

        self.__candidate_history.append(ball_candidates)

        # If all candidates were screened out, meaning there likely was no ball contour we automatically add the
        # prediction as a candidate at current time-step.
        # The above situation can arise due to occlusion (overlapping contours) or ball going out of frame.
        if not ball_candidates:
            self.__candidate_history[-1].extend([prediction])

    def __find_shortest_path_candidate(self) -> Rect:
        """
        Finds the shortest path through sequences of ball candidates.

        A most probable ball candidate is selected based on the idea that a ball's contour size stays almost constant
        and that the movement of a squash ball follows a continuous path.

        :return: Ball candidate at the end of the shortest trajectory through the candidates.
        """
        self.__best_paths.clear()

        # Iterate over the collection of each frame's ball candidates
        # We start from 1 because we don't have anything to compare the first observation against
        for i in range(1, len(self.__candidate_history)):
            # for each candidate Rect in a candidate history "snapshot", assume that the most probable path goes
            # through the candidate
            for point_assumed_best in self.__candidate_history[i]:
                best_dist = sys.maxsize
                best_from_point = None  # Which point was the best to reach 'point_assumed_best'

                # Calculate which was the most probable observation point we came from
                # i.e. this loops over the previous observation layer and compares distances between them.
                for from_point in self.__candidate_history[i - 1]:
                    dist = self.__rect_dist(point_assumed_best, from_point)
                    # Remember the shortest distance and from which point it is achieved
                    if dist < best_dist:
                        best_dist = dist
                        best_from_point = from_point

                if best_from_point in self.__best_paths:
                    # Extend the path
                    self.__best_paths[point_assumed_best] = (self.__best_paths[best_from_point][0] + best_dist,
                                                             best_from_point,
                                                             self.__best_paths[best_from_point][2] + 1)
                else:
                    # Add a start entry for the path
                    self.__best_paths[point_assumed_best] = [best_dist, best_from_point, 1]

        best_dist = sys.maxsize
        best_point = None

        # Iterate over end-of-path rectangles
        for endpoint_rect in self.__best_paths:
            # Only end-Rects that are reached through traversing the full path are considered
            # If this check was not in place, then shorter noisy paths would corrupt the search.
            if self.__best_paths[endpoint_rect][2] == len(self.__candidate_history) - 1:
                dist = self.__best_paths[endpoint_rect][0]

                if best_dist > dist > 2:
                    best_dist = self.__best_paths[endpoint_rect][0]
                    best_point = endpoint_rect
                # If distance throughout the path is very small, then it is likely a non-moving target,
                # thus not the ball
                elif dist <= 2:
                    # print(f"dist: {dist} is < threshsold")
                    # print(f"skipping over {endpoint_rect}")
                    continue
        return best_point

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
        # print(f"INITIAL: Found bounding boxes = {bounding_boxes}")

        bounding_boxes = self.__join_nearby_bounding_boxes(bounding_boxes)
        # print(f"PROCESSED: Found bounding boxes = {bounding_boxes}")

        return [Rect(*rect) for rect in bounding_boxes]

    def __join_nearby_bounding_boxes(self, bounding_boxes: List[list]) -> list:
        """
        :param bounding_boxes: Sorted list of bounding_boxes(rectangles)
        :return: List of rectangles [[x, y, width, height], ...]

        Many thanks to user HansHirse on StackOverflow.
        """

        join_distance = 15
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

    def __get_rectangle_contours(self, rectangle: Rect) -> list:
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

    @staticmethod
    def __rect_dist(rect1: Rect, rect2: Rect) -> float:
        """
        Pseudo-distance function taking into account raw distance and size differences between Rectangles
        :param rect1: Rectangle
        :param rect2: Rectangle
        :return: Custom-'distance' between Rect-s.
        """
        dist = (rect1.x - rect2.x) ** 4 + (rect1.y - rect2.y) ** 4
        if (rect1.width - rect2.width) > 5:
            dist += (rect1.width - rect2.width) ** 4
        else:
            dist += (rect1.width - rect2.width) ** 2
        if (rect1.height - rect2.height) > 5:
            dist += (rect1.height - rect2.height) ** 4
        else:
            dist += (rect1.height - rect2.height) ** 2
        dist = np.sqrt(dist)

        return dist

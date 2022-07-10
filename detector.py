import sys
from operator import itemgetter
from collections import deque

from typing import List

from utils.rect import Rect

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
        self.__candidate_history.append([dummy_candidate])
        # Similarly, means that during frame 2, we also had single ball candidate: 'dummy candidate'
        self.__candidate_history.append([dummy_candidate])

        self.avg_area = 32 * 32  # Experimentally found nice constant
        self.__prev_best_dist = 0
        self.__dist_jump_cutoff = 100

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
        best_candidate = self.__find_shortest_path_candidate(prediction)

        # best_candidate is None in case of no candidates at all -> prediction is
        # selected as the most probable ball candidate.
        if best_candidate is None:
            best_candidate = prediction

        return best_candidate

    def __update_ball_candidates(self, frame: np.ndarray, prediction: Rect) -> None:
        """
        Process contours in the frame and update list of ball candidates from the contours.
        :param prediction: Predicted Rect ball candidate
        :param frame: Binarized video frame containing contours.
        """

        # Reduce noise by joining together nearby contours
        cleaned_contours = self.__join_contours(frame)

        # region DEBUG: Show detector view
        # frame_copy = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        # for contour in cleaned_contours:
        #     utilities.draw_rect(frame_copy, contour, (255, 255, 0))
        # cv.imshow("Detector view", frame_copy)
        # endregion

        # Sort the contours in ascending order based on contour area
        # (Ideally the largest contour is the player and the smallest contour is the ball)
        cleaned_contours.sort(key=lambda rect: rect.area())

        # Filter tiny and excessively large contours
        ball_candidates = list(
            filter(lambda r: 0.5 * self.avg_area <= r.area() <= 3 * self.avg_area, cleaned_contours))

        # Throw away the biggest contour (most likely to be the player) only if such a big contour even exists
        # This prevents the undesirable action of discarding the real ball if it is the largest contour
        if ball_candidates:
            if cleaned_contours[-1].area() > self.avg_area * 1.25:
                ball_candidates = cleaned_contours[:(len(cleaned_contours) - 1)]

        self.__candidate_history.append(ball_candidates)

        # If all candidates were screened out, meaning there likely was no ball contour we automatically add the
        # prediction as a candidate at current time-step.
        # The above situation can arise due to occlusion (overlapping contours) or ball going out of frame.
        if not ball_candidates:
            self.__candidate_history[-1].extend([prediction])

    def __find_shortest_path_candidate(self, prediction) -> Rect:
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
                    dist = np.linalg.norm((point_assumed_best.x - from_point.x, point_assumed_best.y - from_point.y))
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
                # thus not the ball, this avoids locking onto noisy flicker targets.
                elif dist <= 2:
                    continue

        # Avoid sudden jumps in case the ball is lost for a frame or two
        if self.__prev_best_dist < best_dist - self.__dist_jump_cutoff:
            return prediction
        self.__prev_best_dist = best_dist

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
        #https://stackoverflow.com/questions/55376338/how-to-join-nearby-bounding-boxes-in-opencv-python/55385454#55385454
        Algorithm has been adapted for squash-specific use.
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

                    if current_x_max + join_distance >= candxMin:

                        if current_y_min < candyMin:
                            if not current_y_max + 5 >= candyMin:
                                continue
                        else:
                            if not current_y_min - 5 <= candyMax:
                                continue

                        processed[j] = True

                        # Reset coordinates of current rect
                        current_x_max = candxMax
                        current_y_min = min(current_y_min, candyMin)
                        current_y_max = max(current_y_max, candyMax)
                    else:
                        break
                new_bounds.append([current_x_min, current_y_min,
                                   current_x_max - current_x_min, current_y_max - current_y_min])

        return new_bounds

    @staticmethod
    def __get_rectangle_contours(rectangle: list) -> list:
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

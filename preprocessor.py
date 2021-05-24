import logging

import cv2 as cv
from threading import Thread
from queue import Queue
import numpy as np


class Preprocessor:
    """
    The goal of the pre-processing stage of the analysis is to prepare the video for the later stages.
    The key part of this stage is capturing and differentiating the moving parts of the video from the
    static parts of the video.

    To achieve this, the following procedure is used:
    • Three consecutive frames of the video are gathered
    • All three frames are converted to grayscale images
    • Noise reduction is performed using a gaussian filter
    • Consecutive frames are combined via frame differencing
    • The images are combined yet again with a boolean “and” to achieve a single image
    • The image is thresholded to obtain a binary image
    • Morphological operations are used to to enhance and bolster the moving components of the image.

    The result of this procedure will be a binary image with the foreground (moving objects) extracted
    from the background. The foreground consists of exactly our objects of interest - the moving players and the ball.
    """

    def __init__(self, path):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv.VideoCapture(path)
        # initialize the queue used to store frames read from the video file
        self.queue = Queue(maxsize=3)

    def preprocess_frames(self):
        while self.stream.isOpened():
            # .read() is a blocking operation, might want to do something about that in the future
            successful_read, frame = self.stream.read()

            if not successful_read:
                logging.getLogger("Can't receive frame (stream end?). Exiting ...")
                break

            # Covert image to a grayscale image
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # The queue keeps 3 frames in it
            # self.queue.put(gray)

            cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break

        self.stream.release()
        cv.destroyAllWindows()

    def reduce_noise(self):
        pass

    def frame_difference(self):
        pass

    def __gather_initial_frames(self):
        """
        Populates the queue with 3 frames initially to begin the preprocessing stage
        """
        pass

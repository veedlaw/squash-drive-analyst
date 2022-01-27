import timeit

import cv2 as cv
import numpy as np
from collections import deque
from utilities import *
from timeit import default_timer as timer


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

    def __init__(self):
        # deque is convenient, as once at max capacity, it auto-discards the last frame before adding a new one
        self.frame_buffer = deque(maxlen=3)  # contains smoothed grayscale images, used as a "sliding window"
        self.frame_difference_buffer = deque(maxlen=2)  # contains differenced images, used as a "sliding window"
        self.dilation_kernel = np.ones((3, 3), np.uint8)
        self.prev_deflicker = None

    def ready(self) -> bool:
        """
        The preprocessor is ready when the frame buffer is filled.
        :return: True if the buffer has accumulated enough frames to start preprocessing.
        """
        return len(self.frame_buffer) == self.frame_buffer.maxlen

    def add_to_frame_buffer(self, frame: np.ndarray) -> None:
        """
        Prepares the frame and adds it to the frame buffer.

        :param frame: A video frame.
        """
        start = timer()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        end = timer()
        # print(f"grayscaling took {end - start}s")

        start = timer()
        if self.prev_deflicker is None:
            self.prev_deflicker = frame
        else:
            deflickered = self.__deflicker(frame)
            grayscaled = deflickered

        end = timer()
        # print(f"deflickering took {end - start}s")

        start = timer()
        frame = cv.GaussianBlur(frame, (5, 5), 0)
        end = timer()
        # print(f"gaussian blur took {end - start}s")

        self.frame_buffer.append(frame)

        if len(self.frame_buffer) < 3:  # Reading initial frames
            if len(self.frame_buffer) == 2:  # We need two frames to start frame differencing
                self.frame_difference_buffer.append(cv.absdiff(self.frame_buffer[0], self.frame_buffer[1]))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Uses a sliding window approach in the frame buffer for differentiating moving parts of the image from static
        parts.

        Frames are received via the 'frame' parameter and after a few cleaning operations are added to the buffer.

        :return: A binary image that has differentiated moving parts of the image from static parts.
        """

        start = timer()
        # Apply frame differencing to the last two frames
        difference = cv.absdiff(self.frame_buffer[1], self.frame_buffer[2])
        end = timer()
        # print(f"Frame differencing took {end - start}s")
        self.frame_difference_buffer.append(difference)

        # Combine with boolean "AND"
        start = timer()
        combined = cv.bitwise_and(self.frame_difference_buffer[0], self.frame_difference_buffer[1])
        end = timer()
        # print(f"Frame combining took {end - start}s")

        start = timer()
        ret, thresholded = cv.threshold(combined, 0, 255, cv.THRESH_OTSU)
        end = timer()
        # print(f"thresholding took {end - start}s")

        start = timer()
        processed = self.__morphological_close(thresholded, 13)
        end = timer()
        # print(f"Morphological closing took {end - start}s")

        return processed

    def __morphological_close(self, image: np.ndarray, iterations: int) -> np.ndarray:
        """
        Returns the morphological closing (dilation followed by erosion) of the image.

        :param image: Image to apply the operation on.
        :return: The morphological closing of the image
        """

        dilated = cv.dilate(image, self.dilation_kernel, iterations=iterations)
        processed = cv.erode(dilated, self.dilation_kernel)
        return processed

    def __deflicker(self, current_frame: np.ndarray, strengthcutoff=16) -> np.ndarray:
        """
        Compares the corresponding pixels in the last two frames and
        if their difference is below a given threshold, it adjusts the intensity of the
        given pixel in the current frame to be a closer match to the pixel in the previous frame,
        in essence removing some flickering noise.
        :return: Frame with adjusted intensities
        """

        start = timer()
        strength_change_mask = np.abs(current_frame.astype(np.int16) - self.prev_deflicker) < strengthcutoff
        current_frame[strength_change_mask] = self.prev_deflicker[strength_change_mask] + \
                                              np.where(np.greater(current_frame[strength_change_mask],
                                                                  self.prev_deflicker[strength_change_mask]), 1, -1)
        end = timer()
        print(f"Deflicker took {end - start}s")

        self.prev_deflicker = np.copy(current_frame)

        return current_frame

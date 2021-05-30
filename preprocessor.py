import cv2 as cv
import numpy as np
from collections import deque


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
        # deque is convenient, as once at max capacity, it auto-discards the last frame after adding a new one
        self.frame_buffer = deque(maxlen=3)  # contains smoothed grayscale images, used as a "sliding window"
        self.frame_difference_buffer = deque(maxlen=2)  # contains differenced images, used as a "sliding window"
        self.dilation_kernel = np.ones((5, 5), np.uint8)

    def process(self, frame):
        """
        Uses a sliding window approach
        :return: A binary image that differentiates moving parts of the image from static parts from 3 last video frames
        """
        grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        smoothed = cv.GaussianBlur(grayscaled, (5, 5), 0)
        self.frame_buffer.append(smoothed)

        if len(self.frame_buffer) < 3:  # Reading initial frames
            if len(self.frame_buffer) == 2:  # We need two frames to start frame differencing
                self.frame_difference_buffer.append(cv.absdiff(self.frame_buffer[0], self.frame_buffer[1]))
            return None

        # Apply frame differencing to the last two frames
        difference = cv.absdiff(self.frame_buffer[1], self.frame_buffer[2])
        self.frame_difference_buffer.append(difference)

        # Combine with boolean "AND"
        combined = cv.bitwise_and(self.frame_difference_buffer[0], self.frame_difference_buffer[1])

        # Threshold to obtain binary image
        thresholded = cv.threshold(combined, 0, 255, cv.THRESH_OTSU)[1]

        # Morphological closing: dilation -> erosion
        dilated = cv.dilate(thresholded, self.dilation_kernel, iterations=9)  # I played around with this quite a bit and it helped to increase iterations
        processed = cv.erode(dilated, self.dilation_kernel)

        return processed

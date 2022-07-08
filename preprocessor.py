from collections import deque

import cv2

from utilities import *


class Preprocessor:
    """
    Preprocesses video frames and extracts the moving foreground from the static background.
    """
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
        self.__frame_buffer = deque(maxlen=3)  # contains smoothed grayscale images, used as a "sliding window"
        self.__frame_difference_buffer = deque(maxlen=2)  # contains differenced images, used as a "sliding window"
        self.__dilation_kernel = np.ones((3, 3), np.uint8)
        self.__prev_deflicker = None

    def ready(self) -> bool:
        """
        :return: True, if the buffer has been filled and can start preprocessing, False otherwise.
        """
        return len(self.__frame_buffer) == self.__frame_buffer.maxlen

    def initialize_with(self, frame: np.ndarray) -> None:
        """
        Allows populating the frame buffer.
        :param frame: A video frame
        """
        self.__add_to_frame_buffer(frame)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Uses a sliding window approach in the frame buffer for differentiating moving parts of the image from static
        parts.

        Frames are received via the 'frame' parameter and after cleaning operations are added to the buffer.

        :param frame: A video frame
        :return: A binary image that has differentiated moving parts of the image from static parts.
        """
        self.__add_to_frame_buffer(frame)

        # Apply frame differencing to the last two frames
        difference = cv2.absdiff(self.__frame_buffer[1], self.__frame_buffer[2])
        self.__frame_difference_buffer.append(difference)

        # Combine with boolean "AND"
        combined = cv2.bitwise_and(self.__frame_difference_buffer[0], self.__frame_difference_buffer[1])

        # Threshold the combined image
        ret, thresholded = cv2.threshold(combined, 0, 255, cv2.THRESH_OTSU)

        # Dilate the contours via morphological closing
        processed = self.__morphological_close(thresholded, 13)

        return processed

    def __add_to_frame_buffer(self, frame: np.ndarray) -> None:
        """
        Prepares the frame and adds it to the frame buffer.

        :param frame: A video frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.__prev_deflicker is None:
            self.__prev_deflicker = frame
        else:
            self.__deflicker(frame)

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        self.__frame_buffer.append(frame)

        if len(self.__frame_buffer) < 3:  # If just reading initial frames
            if len(self.__frame_buffer) == 2:  # We need two frames to start frame differencing
                self.__frame_difference_buffer.append(cv2.absdiff(self.__frame_buffer[0], self.__frame_buffer[1]))

    def __morphological_close(self, image: np.ndarray, iterations: int) -> np.ndarray:
        """
        Returns the morphological closing (dilation followed by erosion) of the image.

        :param image: Image to apply the operation on.
        :return: The morphological closing of the image
        """

        dilated = cv2.dilate(image, self.__dilation_kernel, iterations=iterations)
        processed = cv2.erode(dilated, self.__dilation_kernel)
        return processed

    def __deflicker(self, current_frame: np.ndarray, strengthcutoff=16) -> None:
        """
        Compares the corresponding pixels in the last two frames and
        if their difference is below a given threshold, it adjusts the intensity of the
        given pixel in the current frame to be a closer match to the pixel in the previous frame,
        in essence removing some flickering noise.
        """

        strength_change_mask = np.abs(current_frame.astype(np.int16) - self.__prev_deflicker) < strengthcutoff
        current_frame[strength_change_mask] = self.__prev_deflicker[strength_change_mask] + \
                                              np.where(np.greater(current_frame[strength_change_mask],
                                                                  self.__prev_deflicker[strength_change_mask]), 1, -1)

        self.__prev_deflicker = np.copy(current_frame)

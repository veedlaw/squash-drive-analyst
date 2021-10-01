import cv2 as cv
import numpy as np
from collections import deque
from utilities import *

from matplotlib import pyplot as plt


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
        self.dilation_kernel = np.ones((3, 3), np.uint8)
        self.prev_deflicker = None

    def process(self, frame, block_thresholds):
        """
        Uses a sliding window approach
        :return: A binary image that differentiates moving parts of the image from static parts from 3 last video frames
        """
        grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 1000 -> allow for change always
        # low -> make changing intensity harder
        # block_thresholds = [1000, 1000, 1000, 8,
        #                     1000, 1000, 1000, 8,
        #                     1000, 8,    8,    8,
        #                     1000, 8,    8,    8,
        #                     1000, 1000, 1000, 8]

        # deflickered = self.deflicker2(grayscaled, block_thresholds)
        # if deflickered is not None:
        #     grayscaled = deflickered

        deflickered = self.deflicker(grayscaled)
        if deflickered is not None:
            grayscaled = deflickered

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

        ret, thresholded = cv.threshold(combined, 0, 255, cv.THRESH_OTSU)
        # thresholded_with_discard = discards_low_intensity_pixels(combined)

        processed = self.morphological_close(thresholded)

        # show_histogram(combined, processed)
        return processed

    def morphological_close(self, image):
        # Morphological closing: dilation -> erosion
        dilated = cv.dilate(image, self.dilation_kernel, iterations=13)
        processed = cv.erode(dilated, self.dilation_kernel)
        return processed

    prev_deflicker = None

    def deflicker(self, frame: np.ndarray, strengthcutoff=16):
        """
        Compares the corresponding pixels in the last two frames and
        if their difference is below a given threshold, it adjusts the intensity of the
        given pixel in the current frame to be a closer match to the pixel in the previous frame,
        in essence removing some flickering noise.
        :return: Frame with adjusted intensities
        """

        if self.prev_deflicker is None:
            self.prev_deflicker = frame
            return

        for row in range(len(frame)):
            for col in range(len(frame[row])):
                    # print(f'i = {i}; row = {row}; col = {col}')

                    prev_intensity = self.prev_deflicker[row, col]
                    curr_intensity = frame[row, col]

                    strength = abs(int(curr_intensity) - int(prev_intensity))
                    # print(f'strength = abs({curr_intensity} - {prev_intensity}) = {strength}')

                    # the strength of the stimulus must be greater than a certain point, else we do not want to allow the
                    # change
                    if strength < strengthcutoff:
                        # print(f'cutoff met: {strength} < {strengthcutoff}')
                        if curr_intensity > prev_intensity:
                            frame[row, col] = prev_intensity + 1
                        else:
                            frame[row, col] = prev_intensity - 1
        self.prev_deflicker = np.copy(frame)
        return frame

    def deflicker2(self, frame: np.ndarray, block_threshold):
        """
        Compares the corresponding pixels in the last two frames and
        if their difference is below a given threshold, it adjusts the intensity of the
        given pixel in the current frame to be a closer match to the pixel in the previous frame,
        in essence removing some flickering noise.
        :return: Frame with adjusted intensities
        """

        if self.prev_deflicker is None:
            self.prev_deflicker = frame
            return

        row = 0
        col = 0
        i = 0
        for image_segment in get_blocks2D(frame):
            width, height = frame.shape[1], frame.shape[0]
            stride_row = int(width / 4)  # magic number TODO
            stride_col = int(height / 5)  # magic number TODO

            if row != 0 and row % 4 == 0:
                row = 0
                col += 1

            strengthcutoff = block_threshold[i]

            for x in range(stride_col):
                x += col * stride_col
                for y in range(stride_row):
                    # print(f'row = {row}; col = {col}; x = {x}; y = {y}')
                    y += row * stride_row

                    prev_intensity = self.prev_deflicker[x, y]
                    curr_intensity = frame[x, y]

                    strength = abs(int(curr_intensity) - int(prev_intensity))
                    # print(f'strength = abs({curr_intensity} - {prev_intensity}) = {strength}')

                    # the strength of the stimulus must be greater than a certain point, else we do not want to allow the
                    # change
                    if strength < strengthcutoff:
                        # print(f'cutoff met: {strength} < {strengthcutoff}')
                        if curr_intensity > prev_intensity:
                            frame[x, y] = prev_intensity + 1
                        elif curr_intensity < prev_intensity:
                            frame[x, y] = prev_intensity - 1
            row += 1
            i += 1
        # self.prev_deflicker = np.copy(frame)
        self.prev_deflicker = frame
        return frame

def show_histogram(img, processed_img):
    """Debugging function that shows plotted histogram of two images."""
    hist_full = cv.calcHist([img], [0], None, [255], [0, 255])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(processed_img, 'gray')
    plt.subplot(223), plt.plot(hist_full)
    plt.xlim([0, 255])
    plt.show()

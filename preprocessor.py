import cv2 as cv
import numpy as np
from collections import deque

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

    def process(self, frame):
        """
        Uses a sliding window approach
        :return: A binary image that differentiates moving parts of the image from static parts from 3 last video frames
        """
        grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # if self.deflicker(grayscaled) is not None:
        #     grayscaled = self.deflicker(grayscaled)

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
        #thresholded_with_discard = discards_low_intensity_pixels(combined)

        processed = self.morphological_close(thresholded)
        #processed_with_discard = self.morphological_close(thresholded_with_discard)

        # show_histogram(combined, processed)
        return processed

    def morphological_close(self, image):
        # Morphological closing: dilation -> erosion
        dilated = cv.dilate(image, self.dilation_kernel, iterations=5)
        processed = cv.erode(dilated, self.dilation_kernel)
        return processed

    prev_deflicker = None
    def deflicker(self, frame: np.ndarray, strengthcutoff=20):
        """Compares the corresponding pixels in the last two frames and
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

                prev_intensity = self.prev_deflicker[row, col]
                curr_intensity = frame[row, col]

                strength = abs(int(curr_intensity) - int(prev_intensity))
                #print(f'strength = abs({curr_intensity} - {prev_intensity}) = {strength}')

                # the strength of the stimulus must be greater than a certain point, else we do not want to allow the
                # change
                if strength < strengthcutoff:
                    #print(f'cutoff met: {strength} < {strengthcutoff}')
                    if curr_intensity > prev_intensity:
                        frame[row, col] = prev_intensity + 1
                    else:
                        frame[row, col] = prev_intensity - 1

        return frame


def discards_low_intensity_pixels(frame):
    # flickering is probably lots of pixels with low intensities
    # discard pixels with low intensities
    # preserves higher intensity pixels
    thresholded1 = cv.threshold(frame, 6, 255, cv.THRESH_TOZERO)[1]
    retf, thresholded = cv.threshold(thresholded1, 0, 255, cv.THRESH_OTSU)
    print("retf: " + str(retf))
    return thresholded

def show_histogram(img, processed_img):
    """Debugging function that shows plotted histogram of two images."""
    hist_full = cv.calcHist([img], [0], None, [255], [0, 255])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(processed_img, 'gray')
    plt.subplot(223), plt.plot(hist_full)
    plt.xlim([0, 255])
    plt.show()

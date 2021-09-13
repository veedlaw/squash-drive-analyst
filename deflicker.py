import logging
import math

import numpy as np
import cv2 as cv


# goal: get local threshold values for subregions of the image for the deflicker method in the preprocessor

# get a ~60 frames spanning histogram for each pixel

# pick some N pixels such that:
#   some have a relatively low mean
#                          medium mean
#                          high mean

# consider how the intensities of picked pixels changes
#   threshold value is derived from the above

# divide the image into some subregions

# pass the information about regions and their thresholds to the deflicker-er

# done.


class Deflicker:
    """
    Contains methods for deriving appropriate local thresholds for de-flickering.
    """

    def __init__(self, video_width, video_height, number_of_incoming_frames=60):
        self.num_frames_read = 0
        self.num_frames_max = number_of_incoming_frames

        self.pixel_intensity_values = np.zeros((video_width, video_height, self.num_frames_max), dtype=np.uint8)
        self.added_pixel_intensity_values = np.zeros((video_width, video_height), dtype=np.uint32)

    def append_pixel_intensity_data(self, frame: np.ndarray):
        """
        Adds for each pixel (in a given frame) its intensity value to the array of pixel_intensity values.
        :param frame: a single video frame
        """

        # Ignore any frames that are out of the bounds of the array
        if self.num_frames_read == self.num_frames_max:
            logging.getLogger("Tried to add frame data beyond pre-allocated memory. Did You set the "
                              "number_of_incoming_frames correctly?")
            return

        grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for x, y in np.ndindex(grayscaled.shape):
            self.pixel_intensity_values[x, y, self.num_frames_read] = grayscaled[x, y]

        self.num_frames_read += 1


    def __calculate_cumulative_mean_intensities(self, array: np.ndarray):
        """
        Calculates the cumulative mean intensity for each pixel in the given array
        :return: Array where each element represents the cumulative intensity of the pixel throughout the given frames.
        """

        num_rows, num_cols, *rest = array.shape
        cumulative_mean_intensities = np.ndarray((num_rows, num_cols))

        for x, y, in np.ndindex((num_rows, num_cols)):
            cumulative_mean_intensities[x, y] = np.mean(array[x, y])
            # print(f'cumulative_intensities[{x}, {y}] = {cumulative_mean_intensities[x,y]}')
        return cumulative_mean_intensities

    def choose_pixels_to_follow(self):

        frame_width = self.pixel_intensity_values.shape[0]
        frame_height = self.pixel_intensity_values.shape[1]

        # By default get_blocks partitions the image into 20 blocks.
        # For each block,
        #   1) We will calculate its mean intensity
        #   2) pick some pixel ~ 25% below mean intensity
        #           some pixel nearest to mean intensity
        #           some pixel ~ 25% above mean intensity
        #      !!! record their indices
        #   3) look at how their intensities changed within the recorded period
        #   ==> derive threshold values from from changes in pixel intensity

        for image_segment in self.get_blocks(self.pixel_intensity_values):
            mean_intensity = self.__calculate_cumulative_mean_intensities(image_segment)
            low_pixel, mean_pixel, high_pixel = self.get_pixels_to_follow_in_block(mean_intensity)
            print(low_pixel)
            print(image_segment[low_pixel])
            print(mean_pixel)
            print(image_segment[mean_pixel])
            print(high_pixel)
            print(image_segment[high_pixel[0], high_pixel[1]])
            break


    def get_pixels_to_follow_in_block(self, arr: np.ndarray):
        """
        Finds pixel-coordinates of 3 pixels:
            1) Intensity closest matches mean_intensity * 0.75
            2) Intensity closest matches mean_intensity
            3) Intensity closest matches mean_intensity * 1.25
        :param arr: 2D-array representing mean intensity of a pixel over a period of time
        :return: Three (x,y) coordinates of pixels:
            1) Intensity closest matches mean_intensity * 0.75
            2) Intensity closest matches mean_intensity
            3) Intensity closest matches mean_intensity * 1.25
        """
        # min = math.inf
        # max = -math.inf

        mean_intensity = np.mean(arr).astype(np.uint8)  # mean intensity in the entire array
        ideal_low_value = mean_intensity * 0.75
        ideal_high_value = mean_intensity * 1.25

        low_pixel_index = None
        low_pixel_intensity = 0
        mean_pixel_index = None
        approx_mean_pixel_intensity = 0
        high_pixel_index = None
        high_pixel_intensity = 0

        for x, y in np.ndindex(arr.shape):
            intensity_value = arr[x, y]
            # if min > intensity_value:
            #     min = intensity_value
            # elif max < intensity_value:
            #     max = intensity_value
            if abs(ideal_low_value - intensity_value) < abs(ideal_low_value - low_pixel_intensity):
                low_pixel_index = (x, y)
                low_pixel_intensity = intensity_value
            if abs(ideal_high_value - intensity_value) < abs(ideal_high_value - high_pixel_intensity):
                high_pixel_index = (x, y)
                high_pixel_intensity = intensity_value
            if abs(mean_intensity - intensity_value) < abs(mean_intensity - approx_mean_pixel_intensity):
                mean_pixel_index = (x, y)
                approx_mean_pixel_intensity = intensity_value

        print(f'mean = {mean_intensity}')
        # print(f'min = {min}')
        # print(f'max = {max}')
        print(f'low pixel intensity: {arr[low_pixel_index]}')
        print(f'mean pixel intensity: {arr[mean_pixel_index]}')
        print(f'high pixel intensity: {arr[high_pixel_index]}')

        return low_pixel_index, mean_pixel_index, high_pixel_index

    def get_blocks(self, arr, num_row_blocks=4, num_col_blocks=5):
        """
        :param arr: Array to partition into blocks
        :return: Generates num_row_blocks * num_col_blocks equally-sized pieces of the array
        """

        height = arr.shape[0]
        width = arr.shape[1]

        stride_row = int(width / num_row_blocks)
        stride_col = int(height / num_col_blocks)

        for x in range(0, arr.shape[0], stride_col):
            for y in range(0, arr.shape[1], stride_row):
                yield arr[x:x + stride_col, y:y + stride_row, :]

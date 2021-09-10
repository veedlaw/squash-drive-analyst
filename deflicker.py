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

    def __init__(self, video_width, video_height):
        self.frames_read = 0

        number_of_frames = 60
        self.pixel_intensity_values = np.zeros((video_width, video_height, number_of_frames), dtype=np.uint8)

        self.added_pixel_intensity_values = np.zeros((video_width, video_height), dtype=np.uint32)

    def add_pixel_intensities(self, frame):
        """
        For each pixel in a frame, adds its value to its cumulative intensity sum.
        """

        grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        for x, y in np.ndindex(frame.shape):
            self.pixel_intensity_values[x, y] += grayscaled[x, y]
        self.frames_read += 1

    def mean_pixel_intensities(self):
        """
        :return: Array of mean pixel intensities
        """
        return self.added_pixel_intensity_values / self.frames_read

    def add_pixel_intensity_data(self, frame):
        """
        Adds for each pixel (in a given frame) its intensity value to the array of pixel_intensity values.
        :param frame: a single video frame
        """
        grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for x, y in np.ndindex(frame.shape):
            self.pixel_intensity_values[x, y, self.frames_read] = grayscaled[x, y]

        self.frames_read += 1
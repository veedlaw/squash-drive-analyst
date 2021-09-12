import cv2 as cv
import logging
import numpy as np


class VideoReader:
    """
    Opens a video capture from a given path and allows for getting video frames.
    """

    def __init__(self, video_path):
        self.__stream = cv.VideoCapture(video_path)
        self.__current_frame_number = 0

    def get_frame(self):
        """
        Feeds video frames using a generator
        :param path: Path to video
        :return: Single frame from video
        """
        while self.__stream.isOpened():
            frame = self.__get_frame_from_stream()
            yield frame

        self.__stream.release()

    def get_N_frames(self, n):
        """
        Gets N frames from the video stream.
        After reading n frames returns the reading position to it's previous position. (frame_count - n)
        :param n: Number of frames to be read
        :return: N frames from the stream as numpy array.
        """
        frame_array = np.empty((n), np.ndarray)

        # stores the current frame the video capture is on
        self.__current_frame_number = self.__stream.get(cv.CAP_PROP_FRAME_COUNT)

        for i in range(n):
            if self.__stream.isOpened():
                frame_array[i] = self.__get_frame_from_stream()

        # Reset reading from previous frame number
        self.set_stream_frame_pos(self.__current_frame_number)

        return frame_array

    def __get_frame_from_stream(self):
        """
        Returns a frame from the class' video stream.
        :return: Read frame, or None if read was unsuccessful
        """
        # .read() is a blocking operation, might want to do something about that in the future
        successful_read, frame = self.__stream.read()

        if not successful_read:
            logging.getLogger("Can't receive frame (stream end?). Exiting ...")

        return frame

    def set_stream_frame_pos(self, stream_pos: int):
        """
        Sets the stream back to the specific frame 'stream_pos'
        :param stream_pos: Number of the frame in the stream
        """
        self.__stream.set(cv.CAP_PROP_POS_FRAMES, stream_pos)

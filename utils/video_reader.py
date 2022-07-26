import cv2 as cv
import logging
import numpy as np
from utils.utilities import FRAME_WIDTH, FRAME_HEIGHT
from threading import Thread
from collections import deque


class VideoReader:
    """
    Opens a video capture from a given path and allows for getting video frames.
    """

    def __init__(self, video_path: str):
        self.__stream = cv.VideoCapture(video_path)
        self.__current_frame_number = 0
        self.__stopped = True
        self.__frame_buffer = deque(maxlen=5)

    def start_reading(self) -> None:
        """
        Starts a producer-thread that fills a buffer with video frames to be read.
        """

        def fill_buf():
            self.__stopped = False
            # Keep reading frames until __stopped or run out of frames to read.
            while not self.__stopped and self.__stream.isOpened():
                if len(self.__frame_buffer) < self.__frame_buffer.maxlen - 1:
                    frame = self.__get_frame_from_stream()
                    if frame is not None:
                        frame_resized = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)
                        self.__frame_buffer.append(frame_resized)
                    else:
                        break
            # Wait for consumer to finish processing remaining frames
            while not self.__stopped and len(self.__frame_buffer) > 0:
                pass
            # Signal end
            self.__stopped = True
            self.__stream.release()

        Thread(target=fill_buf).start()

    def get_frame(self) -> np.ndarray:
        """
        Fetches a video frame from buffer.
        """
        while not self.__stopped:
            if self.__frame_buffer:
                yield self.__frame_buffer.pop()

    def stop_reading(self) -> None:
        """
        Stop the video reader from reading any new frames.
        """
        self.__stopped = True

    def __get_frame_from_stream(self) -> np.ndarray:
        """
        Returns a frame from the class' video __stream.
        :return: Read frame, or None if read was unsuccessful
        """
        successful_read, frame = self.__stream.read()

        if not successful_read:
            # TODO handle in GUI
            logging.getLogger("Can't receive frame (stream end?). Exiting ...")

        return frame


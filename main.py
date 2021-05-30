#!/usr/bin/env python3
from Gui import *
import logging
from preprocessor import *


def main():
    # g = Gui()
    # g.create_and_show_GUI()
    # print(g.file_path) # Debug

    VIDEO_PATH = "resources/test/test_media_normal.mov"
    VIDEO_PATH2 = "resources/test/2players.mp4"
    preprocessor = Preprocessor()

    for frame in get_video_frames(VIDEO_PATH):

        preprocessed = preprocessor.process(frame)
        if preprocessed is not None:
            cv.imshow('frame', preprocessed)  # Debug
            if cv.waitKey(1) == ord('q'):
                break

    get_video_frames(VIDEO_PATH)


def get_video_frames(path):
    """
    Feeds video frames using a generator
    :param path: Path to video
    :return: Single frame from video
    """
    stream = cv.VideoCapture(path)  # Initialize video capturing
    while stream.isOpened():
        # .read() is a blocking operation, might want to do something about that in the future
        successful_read, frame = stream.read()

        if not successful_read:
            logging.getLogger("Can't receive frame (stream end?). Exiting ...")
            break
        yield frame

    stream.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

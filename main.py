#!/usr/bin/env python3
import sys

import cv2
from Gui import *
import logging

from detector import Detector
from preprocessor import *


def main():
    # g = Gui()
    # g.create_and_show_GUI()
    # print(g.file_path) # Debug

    VIDEO_PATH = "resources/test/test_media_normal.mov"
    VIDEO_PATH1 = "resources/test/720p_solo.mov"
    VIDEO_PATH5 = "resources/test/480p_solo.mov"
    VIDEO_PATH3 = "resources/test/2players.mp4"
    VIDEO_PATH2 = "resources/test/rally.mp4"

    preprocessor = Preprocessor()
    #detector = Detector()


    for frame in get_video_frames(VIDEO_PATH5):
        preprocessed = preprocessor.process(frame)

        if preprocessed is not None:

            # RETR external doesn't store any contours within contours
            cnt, hierarchy = cv.findContours(cv.Canny(preprocessed, 0, 1), cv2.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            #
            max_area = sys.maxsize
            index = 0
            i = 0
            for c in cnt:
                area = cv.contourArea(c)
                if area < max_area:
                    index = i
                    max_area = area
                i += 1

            images = []
            #
            img = cv.drawContours(frame, cnt, index, (255, 0, 0), 15)
            cnts = cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            #
            images.append(img)
            images.append(cnts)
            newimg = blend(images)
            cv.imshow('frame', newimg)

            # video.write(processed_without_discard)

            if cv.waitKey(1) == ord('q'):
                break

    # video.release()
    cv.destroyAllWindows()


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


def blend(list_images):  # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output


if __name__ == "__main__":
    main()

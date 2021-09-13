#!/usr/bin/env python3
import sys

import cv2
import numpy as np

from Gui import *
from deflicker import Deflicker

from detector import Detector
from preprocessor import *
from videoReader import VideoReader

# region video paths
VIDEO_PATH = "resources/test/test_media_normal.mov"
VIDEO_PATH1 = "resources/test/720p_solo.mov"
VIDEO_PATH5 = "resources/test/480p_solo.mov"
VIDEO_PATH3 = "resources/test/2players.mp4"
VIDEO_PATH2 = "resources/test/rally.mp4"
# endregion video paths

video_reader = VideoReader(VIDEO_PATH5)


def deflicker_test_method():

    initial_frame = next(video_reader.get_frame())

    if initial_frame is None:
        return

    frame_width = initial_frame.shape[0]
    frame_height = initial_frame.shape[1]
    num_frames_to_read = 30

    deflicker = Deflicker(frame_width, frame_height, num_frames_to_read)
    deflicker.append_pixel_intensity_data(initial_frame)

    for i in range(num_frames_to_read):
        frame = next(video_reader.get_frame())

        if frame is None:
            return

        deflicker.append_pixel_intensity_data(frame)
    deflicker.choose_pixels_to_follow()
    video_reader.set_stream_frame_pos(0)
    print()


def main():
    # region gui
    # g = Gui()
    # g.create_and_show_GUI()
    # print(g.file_path) # Debug
    # endregion gui

    deflicker_test_method()
    # return

    preprocessor = Preprocessor()
    # detector = Detector

    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame in video_reader.get_frame():

        if frame is None:
            return

        preprocessed = preprocessor.process(frame)

        if preprocessed is not None:

            # contoured_image = draw_contours(frame, preprocessed)

            grayscaled = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            print(grayscaled[0, 88])
            mean_intens = np.mean(grayscaled)
            cv2.putText(frame, str(mean_intens), (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

            cv.imshow('frame', frame)

            # video.write(processed_without_discard)

            if cv.waitKey() == ord('q'):
                break

    # video.release()
    cv.destroyAllWindows()


def blend(list_images):  # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output


def draw_contours(frame, preprocessed):
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

    return newimg


if __name__ == "__main__":
    main()

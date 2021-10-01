#!/usr/bin/env python3
import sys
from operator import itemgetter

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
VIDEO_PATH2 = "resources/test/2players.mov"
VIDEO_PATH3 = "resources/test/480pbh.mov"
# endregion video paths

video_reader = VideoReader(VIDEO_PATH5)
preprocessor = Preprocessor()
detector = Detector()


def deflicker_test_method():

    #region initialize
    initial_frame = next(video_reader.get_frame())
    if initial_frame is None:
        return

    frame_width = initial_frame.shape[0]
    frame_height = initial_frame.shape[1]
    num_frames_to_read = 30
    #endregion initialize

    deflicker = Deflicker(frame_width, frame_height, num_frames_to_read)
    deflicker.append_pixel_intensity_data(initial_frame)

    for i in range(num_frames_to_read):
        frame = next(video_reader.get_frame())

        if frame is None:
            return

        deflicker.append_pixel_intensity_data(frame)
    block_thresholds = deflicker.choose_pixels_to_follow()
    video_reader.set_stream_frame_pos(0)

    return block_thresholds

def main():
    # region gui
    # g = Gui()
    # g.create_and_show_GUI()
    # print(g.file_path) # Debug
    # endregion gui

    width = int(video_reader.stream.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video_reader.stream.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    size = (720, 640)
    print(size)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out2 = cv.VideoWriter('test5.avi', fourcc, 20.0, size)

    # block_thresholds = deflicker_test_method()
    block_thresholds = None

    frame_num = 0
    for frame in video_reader.get_frame():

        if frame is None:
            return

        preprocessed = preprocessor.process(frame, block_thresholds)

        if preprocessed is not None:

            # detector.classify(preprocessed)

            contoured_image = draw_contours(frame, preprocessed)
            img = draw_grid(preprocessed)

            # Convert grayscale image to 3-channel image,so that they can be stacked together
            both = np.concatenate((contoured_image, img), axis=1)  # 1 : horz, 0 : Vert.
            cv2.imshow('imgc', both)

            if cv.waitKey() == ord('q'):
                break

            # cv.waitKey(1)

        frame_num += 1
        print(f'frame number = {frame_num}')

    # out2.release()
    cv.destroyAllWindows()


def blend(list_images):  # Blend images equally.

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output



def draw_contours(frame, preprocessed):

    new_rectangles = detector.join_contours(preprocessed)

    # max_area = sys.maxsize
    min_area = sys.maxsize
    for rect in new_rectangles:
        x, y, w, h = rect
        area = w * h
        if area < min_area:
            min_area = area

    for rect in new_rectangles:
        x, y, w, h = rect
        if w * h == min_area:
            img = cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 3)
        else:
            img = cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    # return blend(images)
    return frame


if __name__ == "__main__":
    main()

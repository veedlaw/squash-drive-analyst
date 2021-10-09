#!/usr/bin/env python3
import sys

import cv2

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
VIDEO_PATH3 = "resources/test/short.mov"
# endregion video paths

video_reader = VideoReader(VIDEO_PATH5)
preprocessor = Preprocessor()
detector = Detector()


def main():
    # region gui
    # g = Gui()
    # g.create_and_show_GUI()
    # endregion gui

    width = int(video_reader.stream.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video_reader.stream.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)

    size = (height * 2, width * 2) # *2 for adding images side by side
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out2 = cv.VideoWriter('test5.avi', fourcc, 20.0, size)

    # block_thresholds = utilities.get_deflicker_parameters()
    block_thresholds = None

    frame_num = 0
    for frame in video_reader.get_frame():
        preprocessor.add_to_frame_buffer(frame)

        if preprocessor.ready():
            preprocessed = preprocessor.process(frame, block_thresholds)

            contoured_image = draw_contours(frame, preprocessed, detector)
            img = draw_grid(preprocessed)

            # Convert grayscale image to 3-channel image,so that they can be stacked together
            both = np.concatenate((contoured_image, img), axis=1)  # 1 : horz, 0 : Vert.
            cv2.imshow('imgc', both)

            if cv.waitKey() == ord('q'):
                break

        frame_num += 1
        print(f'frame number = {frame_num}')

    # out2.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

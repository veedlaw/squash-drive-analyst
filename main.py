#!/usr/bin/env python3

import cv2

from detector import Detector
from estimator import Estimator
from preprocessor import *
from videoReader import VideoReader

# region video paths
VIDEO_PATH = "resources/test/480p_solo.mov"
# endregion video paths

video_reader = VideoReader(VIDEO_PATH)
video_reader = VideoReader("../../Downloads/IMG_4189480.mov")
preprocessor = Preprocessor()
estimator = Estimator([0, 0, 0, 0], [0, 0, 0, 0])  # Initially no data about the ball, so initialize with 0-s
detector = Detector()


def initialize_preprocessor():
    for frame in video_reader.get_frame():
        if preprocessor.ready():
            return
        preprocessor.initialize_with(frame)


def main():
    # region gui
    # g = Gui()
    # g.create_and_show_GUI()
    # endregion gui

    debug = False
    cv_frame_wait_time = 1  # 0 for openCV means wait indefinitely

    initialize_preprocessor()

    for frame in video_reader.get_frame():
        preprocessed = preprocessor.process(frame)

        prediction = estimator.predict(t=0.25)
        ball_bounding_box = detector.select_most_probable_candidate(preprocessed, prediction)
        estimator.add_data(position=ball_bounding_box)

        # region drawing
        draw_contour(frame, ball_bounding_box, (0, 0, 255))
        if prediction == ball_bounding_box:
            draw_contour(frame, prediction, (0, 255, 0))
        img = frame
        if debug:
            debug_img = draw_grid(preprocessed)
            img = np.concatenate((frame, debug_img), axis=1)  # Concatenate along horizontal axis
        # endregion drawing

        # region keys
        key = cv2.waitKey(cv_frame_wait_time)
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug = not debug
        elif key == ord('w'):
            cv_frame_wait_time = 0
        elif key == ord('f'):
            cv_frame_wait_time = 1
        # endregion

        cv2.imshow("Video", img)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

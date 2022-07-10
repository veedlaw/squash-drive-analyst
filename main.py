#!/usr/bin/env python3

from bounce_detector import BounceDetector
from detector import Detector
from double_exponential_estimator import DoubleExponentialEstimator
from preprocessor import *
from utils.court import Court
from utils.video_reader import VideoReader

VIDEO_PATH = "../../Downloads/IMG_4189720.mov"

video_reader = VideoReader(VIDEO_PATH)
preprocessor = Preprocessor()
estimator = DoubleExponentialEstimator()
detector = Detector()
bounce_detector = BounceDetector(0, 0)  # TODO dummy initializers for development purposes

court_img = Court.get_court_drawing()
# Court.draw_targets_grid(court_img)

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
    cv_frame_wait_time = 0  # wait value 0 blocks until key press
    cv.imshow("Court View", court_img)
    initialize_preprocessor()

    for frame in video_reader.get_frame():
        preprocessed = preprocessor.process(frame)
        prediction = estimator.predict(t=1)
        if prediction.x < 0 or prediction.y < 0:
            prediction = Rect(-prediction.width, -prediction.height, prediction.width, prediction.height)
        ball_bounding_box = detector.select_most_probable_candidate(preprocessed, prediction)
        estimator.correct(position=ball_bounding_box)

        # region drawing
        draw_rect(frame, prediction, (0, 255, 0))
        draw_rect(frame, ball_bounding_box, (255, 0, 0))
        bounce_detector.show_projection(frame)
        if debug:
            # Show the preprocessed image in parallel to video frame
            frame = np.concatenate((frame, cv.cvtColor(preprocessed, cv.COLOR_GRAY2RGB)), axis=1)
        # endregion drawing

        bounce_detector.update_contour_data(ball_bounding_box)
        if bounce_detector.bounced():
            Court.draw_ball_projection(court_img, *bounce_detector.get_last_bounce_location())
            cv.imshow("Court View", court_img)

        cv2.setMouseCallback("detector_frame", onMouse)
        cv2.setMouseCallback("Projection view", onMouse)
        # region keys
        cv2.imshow("Video", frame)
        cv2.setMouseCallback("Video", onMouse)
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

    cv2.destroyAllWindows()


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x = {x}, y = {y}")


if __name__ == "__main__":
    main()

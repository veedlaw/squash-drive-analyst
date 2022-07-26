import numpy as np

from bounce_detector import BounceDetector
from detector import Detector
from double_exponential_estimator import DoubleExponentialEstimator
from preprocessor import Preprocessor
from stats import create_target_rects, AccuracyStatistics
from utils.court import Court
from utils.rect import Rect
from utils.utilities import draw_rect
from utils.video_reader import VideoReader


class Pipeline:

    def __init__(self, vr: VideoReader, homography_coords: list):

        # Set up the processing pipeline
        self.__video_reader = vr
        self.__preprocessor = Preprocessor()
        self.__estimator = DoubleExponentialEstimator()
        self.__detector = Detector()
        self.__target_rects = create_target_rects()
        self.stats_tracker = AccuracyStatistics(self.__target_rects)
        self.__court_img = Court.get_court_drawing()
        Court.draw_targets_grid(self.__court_img, self.__target_rects)
        self.__bounce_detector = BounceDetector(*homography_coords)

        self.__initialize_preprocessor()

    def process_next(self) -> (np.ndarray, np.ndarray):
        """
        Process the next frame from the video.
        :return: Processed frame and court image
        """
        for frame in self.__video_reader.get_frame():
            processed = self.__process_frame(frame)
            yield processed, self.__court_img

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a single frame
        :param frame: Raw video frame
        :return: Processed frame
        """

        preprocessed = self.__preprocessor.process(frame)
        prediction = self.__estimator.predict(t=1)
        if prediction.x < 0 or prediction.y < 0:
            prediction = Rect(-prediction.width, -prediction.height, prediction.width, prediction.height)
        ball_bounding_box = self.__detector.select_most_probable_candidate(preprocessed, prediction)
        self.__estimator.correct(position=ball_bounding_box)

        # region drawing
        draw_rect(frame, prediction, (0, 255, 0))
        draw_rect(frame, ball_bounding_box, (255, 0, 0))

        self.__bounce_detector.update_contour_data(ball_bounding_box)
        if self.__bounce_detector.bounced():
            x, y = self.__bounce_detector.get_last_bounce_location()
            Court.draw_ball_projection(self.__court_img, x, y)
            # stats_tracker.record_bounce(x, y)
        return frame

    def __initialize_preprocessor(self) -> None:
        """
        Readies the preprocessor by gathering initial frames.
        i.e. handling a boundary case.
        """
        for frame in self.__video_reader.get_frame():
            if self.__preprocessor.ready():
                return
            self.__preprocessor.initialize_with(frame)

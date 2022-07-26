import time
import tkinter as tk

from threading import Thread
import threading

from gui.panel_view import PanelView
from preprocessor import *
from double_exponential_estimator import DoubleExponentialEstimator
from detector import Detector
from bounce_detector import BounceDetector
from stats import AccuracyStatistics, create_target_rects
from utils.court import Court
from utils.video_reader import VideoReader

preprocessor = Preprocessor()
estimator = DoubleExponentialEstimator()
detector = Detector()
bounce_detector = BounceDetector(0, 0)  # TODO dummy initializers for development purposes


class AnalysisView:
    """
    TODO
    """

    def __init__(self, root, init_frame: np.ndarray, video_reader: VideoReader, court_img):
        self.video_reader = video_reader
        self.__root = root
        self.__view = PanelView(root, init_frame)

        self.court_img = court_img
        self.__img = init_frame
        self.__view.update_label_right(court_img)

        self.__debug = False

        root.title(f"Analysing video ...")
        self.__initialize_preprocessor()
        self.__update_event_str = "<<processedFrame>>"

        self.__running = True

        root.bind("<p>", self.__on_pause)

        root.bind(self.__update_event_str, self.__update_view)  # event triggered by background thread

        self.lock = threading.Lock()
        self.cond = threading.Condition(lock=self.lock)
        # Fetch and process frames in separate thread
        Thread(target=self.__run_analysis).start()

    def __run_analysis(self) -> None:
        """TODO"""
        for frame in self.video_reader.get_frame():
            with self.cond:
                while not self.__running:
                    self.cond.wait()
                    self.single_frame = False

                self.__img = self.__process_frame(frame)
                self.__root.event_generate(self.__update_event_str)

    def __update_view(self, event: tk.Event) -> None:
        """Re-draws the frame."""
        self.__view.update_label_left(self.__img)
        self.__view.update_label_right(self.court_img)

    def __on_pause(self, event: tk.Event) -> None:
        """Handles pausing of the processing."""
        self.__running = not self.__running
        # If restarted from pause state notify the processing thread to wake up again
        if self.__running:
            with self.cond:
                self.cond.notify()

    def __process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        TODO
        :param frame:
        :return:
        """

        preprocessed = preprocessor.process(frame)
        prediction = estimator.predict(t=1)
        if prediction.x < 0 or prediction.y < 0:
            prediction = Rect(-prediction.width, -prediction.height, prediction.width, prediction.height)
        ball_bounding_box = detector.select_most_probable_candidate(preprocessed, prediction)
        estimator.correct(position=ball_bounding_box)

        # region drawing
        draw_rect(frame, prediction, (0, 255, 0))
        draw_rect(frame, ball_bounding_box, (255, 0, 0))

        bounce_detector.update_contour_data(ball_bounding_box)
        if bounce_detector.bounced():
            x, y = bounce_detector.get_last_bounce_location()
            # Court.draw_ball_projection(court_img, x, y)
            Court.draw_ball_projection(self.court_img, x, y)
            # stats_tracker.record_bounce(x, y)
        return frame

    def __initialize_preprocessor(self):
        for frame in self.video_reader.get_frame():
            if preprocessor.ready():
                return
            preprocessor.initialize_with(frame)

#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox

from gui import file_selection, set_up_view, guistate
from gui.analysis_view import AnalysisView

from preprocessor import *
from double_exponential_estimator import DoubleExponentialEstimator
from detector import Detector
from bounce_detector import BounceDetector
from stats import AccuracyStatistics, create_target_rects
from utils.court import Court
from utils.video_reader import VideoReader

VIDEO_PATH = "../../Downloads/IMG_4189720.mov"
# VIDEO_PATH = "../../Downloads/bh2.MOV"


class MainApplication(tk.Frame):
    """
    Orchestrates the running of the application.
    """

    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.__master = master

        # Make the view appear in front of other windows when starting the application
        master.lift()
        master.attributes('-topmost', True)
        master.after_idle(root.attributes, '-topmost', False)

        # Go to the file selection view
        self.view = file_selection.FileSelectionView(master)

        # Set up the processing pipeline
        self.video_reader = None  # To-be-selected
        self.preprocessor = Preprocessor()
        self.estimator = DoubleExponentialEstimator()
        self.detector = Detector()
        self.bounce_detector = BounceDetector(0, 0)  # TODO to-be-selected
        self.target_rects = create_target_rects()
        self.stats_tracker = AccuracyStatistics(self.target_rects)
        self.court_img = Court.get_court_drawing()
        Court.draw_targets_grid(self.court_img, self.target_rects)

        master.bind(guistate.SETUP, self.__change_state_SETUP)
        master.bind(guistate.ANALYSIS, self.__change_state_ANALYSIS)

    def __change_state_SETUP(self, evt: tk.Event) -> None:
        """
        Assumes state change from file selection to set-up state.
        State transition fails in case of I/O errors.
        :param evt: TKinter event
        """
        if not self.__try_initialize_video_reader(self.view.file_path):
            return

        self.__initialize_preprocessor()
        self.view = set_up_view.SetUpWindow(root, next(self.video_reader.get_frame()))

    # TODO
    def __change_state_ANALYSIS(self, evt: tk.Event) -> None:
        """
        Assumes state change from set-up to analysis state.
        :param evt: TKinter event
        """
        # Tear down the old frame
        self.view.teardown()
        # Move into analysis view
        self.view = AnalysisView(self.__master, next(self.video_reader.get_frame()), self.video_reader, self.court_img)

    def __try_initialize_video_reader(self, file_path) -> bool:
        """
        :param file_path: Path of video file
        :return: True if successful, False otherwise.
        """
        try:
            self.video_reader = VideoReader(file_path)
            self.video_reader.start_reading()
            return True
        except:
            messagebox.showerror("Error", "An error occurred during file selection.")
            return False

    def __initialize_preprocessor(self) -> None:
        """
        Initializes the PreProcessor with starting frames.
        """
        for frame in self.video_reader.get_frame():
            if self.preprocessor.ready():
                return
            self.preprocessor.initialize_with(frame)

    # TODO
    def __initialize_bounce_detector(self):
        pass




if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()

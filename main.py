#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
import numpy as np

from gui import file_selection, set_up_view, guistate
from gui.analysis_view import AnalysisView
from gui.output_view import OutputView
from pipeline import Pipeline
from stats import AccuracyStatistics
from utils.court import Court

from utils.video_reader import VideoReader


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

        self.__init_frame = None
        self.__video_reader = None  # To-be-selected by user
        self.__stats_tracker = None
        self.__court_img = None

        self.__headless = tk.BooleanVar()
        self.__service_box_dir = tk.IntVar()

        # Bind state transition events.
        master.bind(guistate.SETUP, self.__try_change_state_SETUP)
        master.bind(guistate.ANALYSIS, self.__change_state_ANALYSIS)
        master.bind(guistate.OUTPUT, self.__change_state_OUTPUT)

    def __try_change_state_SETUP(self, evt: tk.Event) -> None:
        """
        Assumes state change from file selection to set-up state.
        State transition fails in case of I/O errors.
        :param evt: TKinter event
        """
        if not self.__try_initialize_video_reader(self.view.file_path):
            return
        self.view.teardown()

        self.__init_frame = next(self.__video_reader.get_frame())
        # Move into setup view state
        self.view = set_up_view.SetUpWindow(root, self.__init_frame, self.__headless, self.__service_box_dir)

    def __change_state_ANALYSIS(self, evt: tk.Event) -> None:
        """
        Assumes state change from set-up to analysis state.
        :param evt: TKinter event
        """

        # Gather necessary coordinates for homography mapping
        service_box_coords_src = np.array(self.view.get_service_box_markers())
        court_lower_coords_src = np.array(self.view.get_court_lower_boundary_coords())
        service_box_coords_dst = Court.get_homography_dst_coords(self.__service_box_dir.get())
        homography_coords = [(service_box_coords_src, court_lower_coords_src), service_box_coords_dst]

        # Tear down the old frame
        self.view.teardown()

        self.__court_img = Court.get_court_drawing()
        self.__stats_tracker = AccuracyStatistics(Court.create_target_rects(self.__service_box_dir.get()))

        pipeline = Pipeline(self.__video_reader, homography_coords, self.__court_img, self.__stats_tracker)
        # Move into analysis view state
        self.view = AnalysisView(self.__master, self.__headless, self.__init_frame, pipeline)

    def __change_state_OUTPUT(self, evt: tk.Event) -> None:
        """
        Assumes state change from Analysis to output state.
        """
        self.view.teardown()
        self.view = OutputView(self.__master, self.__stats_tracker, self.__court_img)

    def __try_initialize_video_reader(self, file_path) -> bool:
        """
        :param file_path: Path of video file
        :return: True if successful, False otherwise.
        """
        try:
            self.__video_reader = VideoReader(file_path)
            self.__video_reader.start_reading()
            return True
        except:
            messagebox.showerror("Error", "An error occurred during file selection.")
            return False


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()
    root.quit()

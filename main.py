#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
from threading import Thread

from gui import file_selection, set_up_view, guistate
from gui.analysis_view import AnalysisView
from pipeline import Pipeline

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
        self.__video_reader = None  # To-be-selected

        master.bind(guistate.SETUP, self.__try_change_state_SETUP)
        master.bind(guistate.ANALYSIS, self.__change_state_ANALYSIS)

    def __try_change_state_SETUP(self, evt: tk.Event) -> None:
        """
        Assumes state change from file selection to set-up state.
        State transition fails in case of I/O errors.
        :param evt: TKinter event
        """
        if not self.__try_initialize_video_reader(self.view.file_path):
            return

        self.__init_frame = next(self.__video_reader.get_frame())
        self.view = set_up_view.SetUpWindow(root, self.__init_frame)

    def __change_state_ANALYSIS(self, evt: tk.Event) -> None:
        """
        Assumes state change from set-up to analysis state.
        :param evt: TKinter event
        """
        # Tear down the old frame
        self.view.teardown()

        pipeline = Pipeline(self.__video_reader, [0, 0])  # TODO
        # Move into analysis view
        self.view = AnalysisView(self.__master, self.__init_frame, pipeline)

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

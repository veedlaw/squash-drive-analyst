import tkinter as tk
import numpy as np

from threading import Thread
import threading

from gui.panel_view import PanelView


class AnalysisView:
    """
    Create and display the analysis view.
    Displays the processed video in parallel with recorded ball bounces
    """

    def __init__(self, master, init_frame: np.ndarray, pipeline):
        self.pipeline = pipeline

        self.__master = master
        self.__view = PanelView(master, init_frame)

        self.court_img = None
        self.__img = init_frame

        self.__debug = False

        self.__master.title(f"Analysing video ...")
        self.__update_event_str = "<<processedFrame>>"

        self.__running = True

        master.bind("<p>", self.__on_pause)
        master.bind(self.__update_event_str, self.__update_view)  # event triggered by background thread

        self.__pause_condition = threading.Condition(lock=threading.Lock())

        # Fetch and process frames in separate thread
        Thread(target=self.__run_analysis).start()

    def __run_analysis(self) -> None:
        """
        Process the analysis pipeline.
        """
        for self.__img, self.court_img in self.pipeline.process_next():
            with self.__pause_condition:
                while not self.__running: self.__pause_condition.wait()
            self.__master.event_generate(self.__update_event_str)

    def __update_view(self, event: tk.Event) -> None:
        """
        Re-draw the frame.
        """

        self.__view.update_label_left(self.__img)
        self.__view.update_label_right(self.court_img)

    def __on_pause(self, event: tk.Event) -> None:
        """
        Handle pausing of the processing.
        """

        self.__running = not self.__running
        # If restarted from pause state notify the processing thread to wake up again
        if self.__running:
            with self.__pause_condition:
                self.__pause_condition.notify()

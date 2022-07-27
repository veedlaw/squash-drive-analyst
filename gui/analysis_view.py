import tkinter as tk
import tkinter.ttk

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

        self.__master = master
        self.__view = PanelView(master, init_frame)

        self.__progress = tk.DoubleVar()
        self.__progress_bar = tkinter.ttk.Progressbar(self.__view.frame, length=master.winfo_width(), variable=self.__progress)
        self.__progress_bar.grid(row=1, columnspan=2)

        self.__master.title(f"Processing video ...")
        self.__update_event_str = "<<processedFrame>>"
        master.bind("<p>", self.__on_pause)
        master.bind(self.__update_event_str, self.__update_view)

        self.__pipeline = pipeline
        self.__court_img = None
        self.__img = init_frame
        self.__running = True
        self.__debug = False

        # Condition to stop analysis processing when application is paused
        self.__pause_condition = threading.Condition(lock=threading.Lock())

        self.__run_analysis()

    def __run_analysis(self) -> None:
        """
        Process the analysis pipeline.
        """
        def run():
            for self.__img, self.__court_img in self.__pipeline.process_next():
                with self.__pause_condition:
                    while not self.__running: self.__pause_condition.wait()
                self.__master.event_generate(self.__update_event_str)

        Thread(target=run, daemon=True).start()

    def __update_view(self, event: tk.Event) -> None:
        """
        Re-draw the frame.
        """

        self.__view.update_label_left(self.__img)
        self.__view.update_label_right(self.__court_img)
        self.__progress.set(self.__pipeline.get_progress() * 100)


    def __on_pause(self, event: tk.Event) -> None:
        """
        Handle pausing of the processing.
        """

        self.__running = not self.__running
        # If restarted from pause state notify the processing thread to wake up again
        if self.__running:
            with self.__pause_condition:
                self.__pause_condition.notify()

from tkinter.constants import LEFT

import numpy as np

from gui.panel_view import PanelView
from stats import AccuracyStatistics


class OutputView:
    """
    Manages the output screen of the application.
    """

    def __init__(self, master, stats: AccuracyStatistics, court: np.ndarray):

        self.__view = PanelView(master, court)

        self.__master = master
        self.__master.title("Processed results")
        self.__stats_tracker = stats
        self.__target_rects = stats.get_target_rects()
        self.__court_img = court

        self.show_stats()

    def show_stats(self) -> None:
        """
        Draw the final output screen.
        """
        # Clear the right label for text
        self.__view.label_right.configure(image='')
        self.__view.label_right.configure(font=("TkDefaultFont", 24), anchor='e', justify=LEFT)

        self.__stats_tracker.draw_box_markings(self.__court_img)
        self.__view.update_label_left(self.__court_img)
        self.__view.label_right.configure(text=(self.__stats_tracker.get_result_str_boxwise()))

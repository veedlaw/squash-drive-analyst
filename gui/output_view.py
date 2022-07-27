import numpy as np

from gui.panel_view import PanelView
from stats import AccuracyStatistics


class OutputView:

    def __init__(self, master, stats: AccuracyStatistics, court: np.ndarray):

        self.__master = master
        self.__stats_tracker = stats
        self.__view = PanelView(master, court)

import tkinter as tk
from tkinter import filedialog
from tkinter.constants import LEFT
import numpy as np

import cv2 as cv
from gui.panel_view import PanelView
from stats import AccuracyStatistics


class OutputView:
    """
    Manages the output screen of the application.
    """

    def __init__(self, master, stats: AccuracyStatistics, court: np.ndarray):

        # GUI setup
        self.__view = PanelView(master, court)
        self.__save_img_button = tk.Button(self.__view.frame, text="Save image", command=self.__on_save_img)
        self.__save_img_button.grid(column=0, row=1, columnspan=2)

        self.__save_txt_button = tk.Button(self.__view.frame, text="Save textual results", command=self.__on_save_txt)
        self.__save_txt_button.grid(column=2, row=1, columnspan=2)

        self.__master = master
        self.__master.title("Processed results")
        self.__stats_tracker = stats
        self.__target_rects = stats.get_target_rects()
        self.__court_img = court

        # String representing results
        self.__text_stats = self.__stats_tracker.get_result_str_boxwise()

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
        self.__view.label_right.configure(text=self.__text_stats)

    def __on_save_img(self) -> None:

        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        cv.imwrite(path, self.__court_img)

    def __on_save_txt(self) -> None:

        path = filedialog.asksaveasfilename(defaultextension=".txt")
        with open(path, 'w') as file:
            file.write(self.__text_stats)


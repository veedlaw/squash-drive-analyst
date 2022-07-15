import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2 as cv
from utils.utilities import FRAME_WIDTH, FRAME_HEIGHT


class Gui:
    """
    Contains methods related to GUI creation and file selection
    """

    def __init__(self):
        self.__mouse_x = FRAME_WIDTH // 2
        self.__mouse_y = FRAME_HEIGHT // 2

    def create_and_show_GUI(self) -> None:
        """
        Creates and displays a window that allows the user to choose a file.
        """

        # Setting up the window
        self.window = tk.Tk()
        self.window.title("Squash Analysis")
        select_file_button = tk.Button(self.window, text="Choose File",
                                       command=self.select_video_path).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Makes the window appear in front of other windows when starting the application
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after_idle(self.window.attributes, '-topmost', False)
        self.window.mainloop()

    def select_video_path(self) -> None:
        """
        Opens a file choosing dialog, allowing the user to select a video file.
        """

        self.file_path = filedialog.askopenfilename(
            title="Choose a file",
            filetypes=[("video files", (".mp4", ".mov", ".MOV"))]
        )
        self.window.destroy()

    def show_box_selection_window(self, img: np.ndarray) -> list:
        """
        Show a window to select court boundaries with a magnifying glass effect.
        :param img: Initial frame
        :return: TODO
        """
        WINDOW_NAME = "Select required points"
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)

        window_size_x = FRAME_WIDTH // 10
        window_size_x_half = window_size_x // 2
        window_size_y = FRAME_HEIGHT // 10
        window_size_y_half = window_size_y // 2

        while True:
            cv.setMouseCallback(WINDOW_NAME, self.onMouse)

            # Handling mouse at window boundary cases also:
            from_x = max(min(self.__mouse_x, FRAME_WIDTH) - (window_size_x_half - 1), window_size_x_half - 1)
            to_x = min(self.__mouse_x + (window_size_x_half + 1), FRAME_WIDTH - (window_size_x_half - 3))
            from_y = max(self.__mouse_y - (window_size_y_half - 1), window_size_y_half)
            to_y = min(self.__mouse_y + (window_size_y_half + 1), FRAME_HEIGHT - (window_size_y_half - 1))

            # "window" pixels
            pixels = img[from_y:to_y, from_x:to_x]
            magnified = cv.resize(pixels, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)

            cv.rectangle(magnified, (magnified.shape[1] // 2, magnified.shape[0] // 2),
                         (magnified.shape[1] // 2 + 5, magnified.shape[0] // 2 + 10), (0, 0, 0), 1)

            img_copy = np.concatenate((img, magnified), axis=1)

            cv.imshow(WINDOW_NAME, img_copy)

            if cv.waitKey(1) == ord('q'):
                cv.destroyWindow(WINDOW_NAME)
                break

    def onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.__mouse_x = x
            self.__mouse_y = y

        self.onClick(event, x, y, flags, param)

    def onClick(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"x = {x}, y = {y}")

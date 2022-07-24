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
        # Initially set mouse position at the center of the frame
        self.__mouse_x = FRAME_WIDTH // 2
        self.__mouse_y = FRAME_HEIGHT // 2
        # Stores clicked points
        self.__markers = []

        # Magnifying "window" size
        self.window_size_x_half = FRAME_WIDTH // 10 // 2
        self.window_size_y_half = FRAME_HEIGHT // 10 // 2


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
        num_box_coords = 4

        while True:
            cv.setMouseCallback(WINDOW_NAME, self.__onMouse)

            img_copy = np.copy(img)
            for count, (x, y) in enumerate(self.__markers):
                cv.rectangle(img_copy, (x, y), (x + 2, y + 2), (0, 0, 255), -1)

                if count == 0:
                    continue
                elif count == num_box_coords - 1:
                    cv.line(img_copy, (x + 1, y + 1), (self.__markers[0][0] + 1, self.__markers[0][1] + 1),
                            (0, 0, 255), 1, lineType=cv.LINE_AA)
                if count < num_box_coords:
                    cv.line(img_copy, (self.__markers[count-1][0] + 1, self.__markers[count-1][1] + 1),
                            (x + 1, y + 1), (0, 0, 255), 1, lineType=cv.LINE_AA)

            from_x, to_x, diff_x, from_y, to_y, diff_y = self.__get_magnifying_coordinates()

            # "window" pixels
            pixels = img_copy[from_y:to_y, from_x:to_x]
            magnified = cv.resize(pixels, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)
            # Draw magnifying cursor
            cursor_offset = (diff_x * 9, diff_y * 9)
            rect_cursor_start = np.add((magnified.shape[1] // 2, magnified.shape[0] // 2), cursor_offset)
            rect_cursor_end = np.add((magnified.shape[1] // 2 + 5, magnified.shape[0] // 2 + 10), cursor_offset)
            cursor_color = cv.bitwise_not(magnified[rect_cursor_start[1] - cursor_offset[1],
                                                    rect_cursor_start[0] - cursor_offset[0]]).flatten().tolist()
            cv.rectangle(magnified, rect_cursor_start, rect_cursor_end, cursor_color, 1)
            # Concatenate original and magnified view to show side by side
            cv.imshow(WINDOW_NAME, np.concatenate((img_copy, magnified), axis=1))

            key = cv.waitKey(1)
            if key == ord('q'):
                cv.destroyWindow(WINDOW_NAME)
                break
            # 'u' for Undo
            elif key == ord('u'):
                if self.__markers:
                    self.__markers.pop()

    def __get_magnifying_coordinates(self) -> (int, int, int, int, int, int):
        """
        :return: Start- and end-range of the magnifying window with cursor-adjusted diff_ variables.
        """

        from_x = self.__mouse_x - self.window_size_x_half - 1
        to_x = self.__mouse_x + self.window_size_x_half + 1
        from_y = self.__mouse_y - self.window_size_y_half - 1
        to_y = self.__mouse_y + self.window_size_y_half + 1

        # Handling boundary cases
        diff_y = 0
        if to_y >= FRAME_HEIGHT:
            diff_y = to_y - FRAME_HEIGHT
        elif from_y <= 0:
            diff_y = from_y
            from_y = 0

        diff_x = 0
        if to_x >= FRAME_WIDTH:
            diff_x = to_x - FRAME_WIDTH
            if to_x - FRAME_WIDTH >= self.window_size_x_half + 1:
                from_x = FRAME_WIDTH - self.window_size_x_half - 1
                to_x = FRAME_WIDTH
        elif from_x <= 0:
            diff_x = from_x
            from_x = 0

        return from_x, to_x, diff_x, from_y, to_y, diff_y

    def __onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.__mouse_x = x
            self.__mouse_y = y

        self.__onClick(event, x, y, flags, param)

    def __onClick(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Only allow a maximum of 6 points the list.
            if len(self.__markers) < 6:
                self.__markers.append((x, y))

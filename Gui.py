import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import cv2 as cv
import PIL.Image, PIL.ImageTk
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
        self.__num_back_court_coords = 2
        self.__num_box_coords = 4
        self.__NUM_MARKERS_REQUIRED = self.__num_box_coords + self.__num_back_court_coords
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

    def create_magnifying_selection_window(self, frame: np.ndarray):
        self.window = tk.Tk()

        self.__WINDOW_TITLE_BASE = "Select required points "
        self.window.title(f"{self.__WINDOW_TITLE_BASE}: 0/{self.__NUM_MARKERS_REQUIRED}")

        # open video source (by default this will try to open the computer webcam)
        # self.vid = MyVideoCapture(self.video_source)
        self.img = frame
        self.img_TK = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv.cvtColor(self.img, cv.COLOR_BGR2RGB)))

        self.full_img_label = tk.Label(image=self.img_TK)
        self.full_img_label.grid(column=0, row=0, padx=10, pady=10)
        self.magnified_label = tk.Label(image=self.img_TK)
        self.magnified_label.grid(column=1, row=0, padx=10, pady=10)

        self.__undo_button=tk.Button(self.window, text="Undo", width=50, command=self.__on_undo)
        self.__undo_button.grid(column=0, columnspan=2)

        self.window.bind('<Motion>', self.__on_motion)
        self.window.bind("<Button-1>", self.__on_click)
        self.window.mainloop()

    def __update_zoom(self) -> None:
        """
        Draw the zoomed-in view of the full image.
        """
        self.img_copy = np.copy(self.img)
        self.__draw_markers(self.img_copy)
        from_x, to_x, diff_x, from_y, to_y, diff_y = self.__get_magnifying_coordinates()

        # "window" pixels
        pixels = self.img_copy[from_y:to_y, from_x:to_x]
        magnified = cv.resize(pixels, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)
        # Draw magnifying cursor
        cursor_offset = (diff_x * 9, diff_y * 9)
        rect_cursor_start = np.add((magnified.shape[1] // 2, magnified.shape[0] // 2), cursor_offset)
        rect_cursor_end = np.add((magnified.shape[1] // 2 + 5, magnified.shape[0] // 2 + 10), cursor_offset)
        cursor_color = cv.bitwise_not(magnified[rect_cursor_start[1] - cursor_offset[1],
                                                rect_cursor_start[0] - cursor_offset[0]]).flatten().tolist()
        cv.rectangle(magnified, rect_cursor_start, rect_cursor_end, cursor_color, 1)

        self.__magnified = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv.cvtColor(magnified, cv.COLOR_BGR2RGB)))
        self.magnified_label.configure(image=self.__magnified)

    def __update_full_img_label(self) -> None:
        """
        Re-draws the full-image label.
        """
        self.panel1_tk_img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv.cvtColor(self.img_copy, cv.COLOR_BGR2RGB)))
        self.full_img_label.configure(image=self.panel1_tk_img)

    def __get_magnifying_coordinates(self) -> (int, int, int, int, int, int):
        """
        :return: Start- and end-range of the magnifying window with cursor-adjusted diff_ variables.
        """
        # Create from and -to indices of the window
        from_x = self.__mouse_x - self.window_size_x_half - 1
        to_x = self.__mouse_x + self.window_size_x_half + 1
        from_y = self.__mouse_y - self.window_size_y_half - 1
        to_y = self.__mouse_y + self.window_size_y_half + 1

        # Handle boundary cases
        diff_y = 0
        if to_y >= FRAME_HEIGHT:
            diff_y = to_y - FRAME_HEIGHT
        elif from_y <= 0:
            diff_y = from_y
            from_y = 0

        diff_x = 0
        if to_x >= FRAME_WIDTH:
            diff_x = to_x - FRAME_WIDTH
        elif from_x <= 0:
            diff_x = from_x
            from_x = 0

        return from_x, to_x, diff_x, from_y, to_y, diff_y

    def __draw_markers(self, frame) -> None:
        """
        Draws box outline of selection on frame.
        :param frame: Frame to draw on
        """
        for count, (x, y) in enumerate(self.__markers):
            # Draw the marker rectanle
            cv.rectangle(frame, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), -1)

            if count == 0:
                continue
            # Draw joining lines between the markers
            elif count == self.__num_box_coords - 1:
                cv.line(frame, (x + 1, y + 1), (self.__markers[0][0] + 1, self.__markers[0][1] + 1),
                        (0, 0, 255), 1, lineType=cv.LINE_AA)
            if count < self.__num_box_coords:
                cv.line(frame, (self.__markers[count - 1][0] + 1, self.__markers[count - 1][1] + 1),
                        (x + 1, y + 1), (0, 0, 255), 1, lineType=cv.LINE_AA)

            # Draw a joining line between the last two markers
            if count == self.__NUM_MARKERS_REQUIRED - 1:
                cv.line(frame, (self.__markers[count - 1][0] + 1, self.__markers[count - 1][1] + 1),
                        (x + 1, y + 1), (0, 0, 255), 1, lineType=cv.LINE_AA)

    def __on_undo(self) -> None:
        """
        Handles undo button click.
        """
        if self.__markers:
            self.__markers.pop()
            self.__update_zoom()
            self.__update_full_img_label()

    def __on_click(self, event) -> None:
        """
        Handle mouse clicks and re-draw the window
        :param event: TKinter event
        """
        self.__update_mouse_pos()
        print(f'x = {self.__mouse_x}')
        print(f'y = {self.__mouse_y}')

        # Ignore clicks outside the selection area of markers
        if not self.__mouse_within_bounds():
            return

        # Limit the size of the markers list
        if len(self.__markers) < self.__NUM_MARKERS_REQUIRED:
            self.__markers.append((self.__mouse_x, self.__mouse_y))

        self.__update_zoom()
        self.__update_full_img_label()
        self.window.title(f"{self.__WINDOW_TITLE_BASE}: {len(self.__markers)}/{self.__NUM_MARKERS_REQUIRED}")

    def __on_motion(self, event) -> None:
        """
        Update mouse position and re-draw zoom window.
        :param event: TKinter event
        """
        self.__update_mouse_pos()
        if self.__mouse_within_bounds():
            self.__update_zoom()

    def __update_mouse_pos(self) -> None:
        """
        Update the mouse position relative to the marker clicking area.
        """
        self.__mouse_x = self.full_img_label.winfo_pointerx() - self.full_img_label.winfo_rootx()
        self.__mouse_y = self.full_img_label.winfo_pointery() - self.full_img_label.winfo_rooty()

    def __mouse_within_bounds(self) -> bool:
        """
        :return: Whether the last reading of the mouse position lies within the boundaries of the frame.
        """
        return 0 <= self.__mouse_x <= FRAME_WIDTH and 0 <= self.__mouse_y <= FRAME_HEIGHT

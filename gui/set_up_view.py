import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import messagebox

from gui import guistate
from gui.panel_view import PanelView
from utils.utilities import FRAME_WIDTH, FRAME_HEIGHT


class SetUpWindow:
    """
    Handles the pre-analysis set-up window of the application and transfers control to the analysis once
    the user confirms readiness.
    """

    def __init__(self, master, init_frame: np.ndarray, headless_var: tk.BooleanVar):

        self.__master = master
        self.__headless = headless_var

        # GUI setup
        self.__view = PanelView(master, init_frame)
        self.__undo_button = tk.Button(self.__view.frame, text="Undo", width=10, command=self.__on_undo)
        self.__undo_button.grid(column=0, row=1)

        self.__checkbutton = tk.Checkbutton(self.__view.frame, variable=self.__headless,
                text="Show processing video (Slower)", onvalue=False, offvalue=True)
        self.__checkbutton.grid(row=1, column=1, sticky='W')

        self.__img = init_frame
        self.__img_copy = None

        # Stores clicked points
        self.__markers = []

        # Magnified "view" size
        self.window_size_x_half = FRAME_WIDTH // 10 // 2
        self.window_size_y_half = FRAME_HEIGHT // 10 // 2

        self.__num_back_court_coords = 2
        self.__num_box_coords = 4
        self.__NUM_MARKERS_REQUIRED = self.__num_box_coords + self.__num_back_court_coords

        self.__WINDOW_TITLE_BASE = "Select required points "
        master.title(f"{self.__WINDOW_TITLE_BASE}: 0/{self.__NUM_MARKERS_REQUIRED}")

        # Initially set mouse position at the center of the frame
        self.__mouse_x = FRAME_WIDTH // 2
        self.__mouse_y = FRAME_HEIGHT // 2

        self.__binds = {'<Motion>': self.__on_motion, '<Button-1>': self.__on_click}
        for evt, func in self.__binds.items():
            master.bind(evt, func)

    def __show_start_analysis_dialog(self) -> None:
        """
        Show a dialog for progressing into the analysis of the video.
        Upon clicking ok progresses into video analysis.
        """
        if messagebox.askokcancel("Start analysis?", "All required points have been selected. Start analysis?"):
            self.__master.event_generate(guistate.ANALYSIS)

    def __update_zoom(self) -> None:
        """
        Draw the zoomed-in view of the full image.
        """
        self.__img_copy = np.copy(self.__img)
        self.__draw_markers(self.__img_copy)
        from_x, to_x, diff_x, from_y, to_y, diff_y = self.__get_magnifying_coordinates()

        # "window" pixels
        pixels = self.__img_copy[from_y:to_y, from_x:to_x]
        magnified = cv.resize(pixels, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)
        # Draw magnifying cursor
        cursor_offset = (diff_x * 9, diff_y * 9)
        rect_cursor_start = np.add((magnified.shape[1] // 2, magnified.shape[0] // 2), cursor_offset)
        rect_cursor_end = np.add((magnified.shape[1] // 2 + 5, magnified.shape[0] // 2 + 10), cursor_offset)
        cursor_color = cv.bitwise_not(magnified[rect_cursor_start[1] - cursor_offset[1],
                                                rect_cursor_start[0] - cursor_offset[0]]).flatten().tolist()
        cv.rectangle(magnified, rect_cursor_start, rect_cursor_end, cursor_color, 1)

        self.__view.update_label_right(magnified)

    def __draw_markers(self, frame) -> None:
        """
        Draws box outline of selection on frame.
        :param frame: Frame to draw on
        """
        for count, (x, y) in enumerate(self.__markers):
            # Draw the marker rectangle
            cv.rectangle(frame, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), -1)

            if count == 0:
                # Need at least two points two draw a line
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

    def __get_magnifying_coordinates(self) -> (int, int, int, int, int, int):
        """
        :return: Start- and end-range of the magnifying view with cursor-adjusted diff_ variables.
        """
        # Create from and -to indices of the view
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

    def __on_click(self, event: tk.Event) -> None:
        """
        Handle mouse clicks and re-draw both normal and zoom view.
        :param event: TKinter event
        """
        self.__update_mouse_pos()

        # Ignore clicks outside the selection area of markers
        if not self.__mouse_within_bounds():
            return

        # Limit the size of the markers list
        if len(self.__markers) < self.__NUM_MARKERS_REQUIRED:
            self.__markers.append((self.__mouse_x, self.__mouse_y))

        self.__update_zoom()
        self.__view.update_label_left(self.__img_copy)
        self.__update_title()
        self.__master.update()

        if len(self.__markers) == self.__NUM_MARKERS_REQUIRED:
            self.__show_start_analysis_dialog()

    def __on_motion(self, event: tk.Event) -> None:
        """
        Update mouse position and re-draw zoom view.
        :param event: TKinter event
        """
        self.__update_mouse_pos()
        if self.__mouse_within_bounds():
            self.__update_zoom()

    def __on_undo(self) -> None:
        """
        Handles undo button click.
        """

        if self.__markers:
            self.__mouse_x = 180  # TODO HACK
            self.__mouse_y = 180  # TODO HACK

            self.__markers.pop()
            self.__update_zoom()
            self.__view.update_label_left(self.__img_copy)
            self.__update_title()

    def __update_mouse_pos(self) -> None:
        """
        Update the mouse position relative to the marker clicking area.
        """
        self.__mouse_x, self.__mouse_y = self.__view.mouse_pos_wrt_left_label()

    def __mouse_within_bounds(self) -> bool:
        """
        :return: Whether the last reading of the mouse position lies within the boundaries of the frame.
        """
        return 0 <= self.__mouse_x <= FRAME_WIDTH and 0 <= self.__mouse_y <= FRAME_HEIGHT

    def __update_title(self) -> None:
        """
        Update application titlebar
        """
        self.__master.title(f"{self.__WINDOW_TITLE_BASE}: {len(self.__markers)}/{self.__NUM_MARKERS_REQUIRED}")

    def teardown(self) -> None:
        """
        Destroys the SetUpWindow frame and unbinds all events.
        """
        self.__view.teardown()
        self.__master.title("")
        for event in self.__binds:
            self.__master.unbind(event)

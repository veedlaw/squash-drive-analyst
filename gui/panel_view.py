import tkinter as tk
import PIL.Image, PIL.ImageTk
import numpy as np
import cv2 as cv


class PanelView(tk.Frame):
    """
    Handles creation and management of a 2-panel side-by-side view.
    """

    def __init__(self, master, init_frame: np.ndarray):
        tk.Frame.__init__(self, master)
        self.__master = master

        # Create the frame
        self.frame = tk.Frame(master)
        self.frame.grid()

        # Create the labels that hold the images
        self.label_left = tk.Label(self.frame)
        self.label_left.grid(column=0, row=0, padx=10, pady=10, columnspan=2)
        self.label_right = tk.Label(self.frame)
        self.label_right.grid(column=2, row=0, padx=10, pady=10, columnspan=2)

        self.panel1_tk_img = None
        self.panel2_tk_img = None
        self.update_label_left(init_frame)
        self.update_label_right(init_frame)

    def update_label_left(self, frame: np.ndarray) -> None:
        """
        Update image on left PanelView label.
        :param frame: OpenCV image
        """
        self.panel1_tk_img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
        self.label_left.configure(image=self.panel1_tk_img)

    def update_label_right(self, frame: np.ndarray) -> None:
        """
        Update image on right PanelView label.
        :param frame: OpenCV image
        """
        self.panel2_tk_img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
        self.label_right.configure(image=self.panel2_tk_img)

    def mouse_pos_wrt_left_label(self) -> (int, int):
        """
        :return: Mouse position with respect to the coordinate system of the left label of PanelView.
        """
        return self.label_left.winfo_pointerx() - self.label_left.winfo_rootx(), \
               self.label_left.winfo_pointery() - self.label_left.winfo_rooty()

    def teardown(self) -> None:
        """Destroys the created frame"""
        self.frame.destroy()

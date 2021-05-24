import tkinter as tk
from tkinter import filedialog


class Gui:
    """
    Contains methods related to GUI creation and file selection
    """

    def create_and_show_GUI(self):
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

    def select_video_path(self):
        """
        Opens a file choosing dialog, allowing the user to select a video file.
        """

        self.file_path = filedialog.askopenfilename(
            title="Choose a file",
            filetypes=[("video files", (".mp4", ".mov", ".MOV"))]
        )
        self.window.destroy()

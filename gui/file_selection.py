import tkinter as tk
from tkinter import filedialog

from gui import guistate


class FileSelectionView:
    """Create and display a view that allows the user to select a video file."""

    def __init__(self, master):

        self.__master = master
        master.title("Select a video")
        select_file_button = tk.Button(master, text="Choose File", command=self.select_video_path)
        select_file_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.file_path = None

    def select_video_path(self) -> None:
        """
        Opens a file choosing dialog, allowing the user to select a video file.
        """

        self.file_path = filedialog.askopenfilename(
            title="Choose a file",
            filetypes=[("video files", (".mp4", ".mov", ".MOV"))]
        )

        # If a file has been selected try to transfer to the SETUP state.
        if self.file_path:
            self.__master.event_generate(guistate.SETUP, data=self.file_path)

    def teardown(self) -> None:
        self.__master.title("")
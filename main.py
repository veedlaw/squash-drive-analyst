#!/usr/bin/env python3
from Gui import *
from preprocessor import *


def main():
    # g = Gui()
    # g.create_and_show_GUI()
    # print(g.file_path) # Debug

    VIDEO_PATH = "resources/test/test_media_normal.mov"
    preprocessor = Preprocessor(VIDEO_PATH)
    preprocessor.show_video()


if __name__ == "__main__":
    main()

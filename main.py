#!/usr/bin/env python3
from Gui import *

class Main:
    def main():
        g = Gui()
        g.createAndShowGUI()
        print(g.file_path) # Debug

    if __name__ == "__main__":
        main()
#!/usr/bin/env python3
import os
import tkinter as tk
from typing import Callable, Tuple, Any
from tkinter import filedialog
from threading import Thread, Event
import datetime

from record import record
from utils import parse_timestamp
from config import Config

config = Config("config.json")


class App:
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the tkinter app with a root widget.

        Args:
        root: The root widget for the tkinter application.
        """
        self.root = root
        self.is_recording: bool = False  # Flag to keep track of recording state

        # Create start/stop button
        self.button: tk.Button = tk.Button(
            root, text="Start Recording", command=self.toggle_recording
        )
        self.button.pack()

        # Create entry for file name
        self.filename: tk.Entry = tk.Entry(root)
        self.filename.pack()
        self.filename.insert(0, "default_name")

        # Create button to select file path
        self.path_button: tk.Button = tk.Button(
            root, text="Select Save Path", command=self.select_path
        )
        self.path_button.pack()

        # Label to display selected path
        self.path_label: tk.Label = tk.Label(root, text="")
        self.path_label.pack()

        # select preconfigured file path if set
        if config["save_path"]:
            self.select_path(config["save_path"])

        self.stop_event = Event()

    def toggle_recording(self) -> None:
        """
        Toggle the recording state. Changes the text on the recording button and prints the current recording state
        and the entered file name. Replace the print statements with your start/stop recording logic.
        """
        if self.is_recording:
            # stop recording
            self.button.config(text="Start Recording")
            self.filename.config(state="normal")
            self.is_recording = False
            self.stop_event.set()

        else:
            # start recording
            self.button.config(text="Stop Recording")
            self.is_recording = True
            self.filename.config(state="disabled")
            file_path = os.path.join(
                config["save_path"],
                f"{self.filename.get()}_{parse_timestamp(str( datetime.datetime.utcnow() )).strftime('%Y%m%d_%H%M%S')}.mp3",
            )
            self.stop_event.clear()
            Thread(
                target=record,
                args=(
                    self.stop_event,
                    file_path,
                ),
            ).start()

        print(f"Recording: {self.is_recording}")
        print(f"File Name: {self.filename.get()}")

    def select_path(self) -> None:
        """
        Open a file dialog to select a directory and update the path label with the selected path.
        """
        config["save_path"] = filedialog.askdirectory()
        self.path_label.config(text=f"Selected Path: {config['save_path']}")


if __name__ == "__main__":
    root = tk.Tk()
    app: App = App(root)
    root.mainloop()

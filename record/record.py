import sounddevice as sd
import numpy as np
from datetime import datetime
from utils import parse_timestamp
from typing import Callable, Tuple, Any
from pydub import AudioSegment
from typing import List
from threading import Thread, Event
import eyed3
import time

RATE = 44100


def record_audio(recorded_data: List[np.ndarray], stop_event: Event) -> None:
    """
    Start recording audio from the default input device using the global sampling rate.
    Recording is stopped when the user presses Enter.

    Args:
    recorded_data: A list that will be filled with the recorded data chunks.
    """

    def audio_callback(
        indata: np.ndarray, frames: int, time: Tuple[int, int], status: sd.CallbackFlags
    ) -> None:
        """
        Callback function for the audio recording. This is called for every chunk of audio data that is recorded.

        Args:
        indata: A two-dimensional NumPy array with the recorded input signal.
        frames: The number of frames in `indata`.
        time: A two-element tuple consisting of the input and output latency in seconds.
        status: An instance of CallbackFlags indicating whether an underflow, overflow or other special condition has occurred.
        """
        recorded_data.append(indata.copy())

    with sd.InputStream(samplerate=RATE, channels=1, callback=audio_callback):
        print("Recording started... Press Enter to stop recording.")
        while not stop_event.is_set():
            time.sleep(0.1)
        print("Recording stopped.")


def save_audio_with_timestamps(start_timestamp, end_timestamp, recorded_data):
    start_timestamp = parse_timestamp(start_timestamp)
    end_timestamp = parse_timestamp(end_timestamp)

    audio_data = np.concatenate(recorded_data, axis=0)

    audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

    # Convert the NumPy array to a Pydub AudioSegment
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=RATE,
        sample_width=audio_data.dtype.itemsize,
        channels=1,
    )

    # Export the audio file without metadata
    filename = f"audio_{start_timestamp.strftime('%Y%m%d_%H%M%S')}_to_{end_timestamp.strftime('%Y%m%d_%H%M%S')}.mp3"
    audio.export(filename, format="mp3")

    # Set the metadata using eyed3
    mp3file = eyed3.load(filename)
    if mp3file.tag is None:
        mp3file.initTag()

    mp3file.tag.comments.set(
        f"{start_timestamp.isoformat()}\n{end_timestamp.isoformat()}"
    )
    mp3file.tag.save()

    print(f"Audio saved as {filename}")


def record(stop_event: Event) -> None:
    """
    Start recording audio data until the stop_event is set. Then, save the recorded data with start
    and end timestamps.

    Args:
    stop_event: The Event that signals when to stop recording.
    """
    # Initialize a list to store the recorded data
    recorded_data: List[np.ndarray] = []

    # Set the sampling rate and the number of channels
    sampling_rate: int = 44100
    num_channels: int = 1

    # Get the current UTC timestamp as the start timestamp
    start_timestamp: str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Start recording audio data. The recording will be stopped when the stop_event is set.
    record_audio(recorded_data, stop_event)

    print("\nRecording stopped.")

    # Get the current UTC timestamp as the end timestamp
    end_timestamp: str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Save the recorded data with the start and end timestamps
    save_audio_with_timestamps(start_timestamp, end_timestamp, recorded_data)

    print("Audio saved with start and end timestamps.")


if __name__ == "__main__":
    se = Event()
    Thread(target=record, args=(se,)).start()
    input()
    se.set()

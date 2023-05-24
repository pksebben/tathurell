#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from datetime import datetime
import struct
import io
import sys
import keyboard
import wavio
import eyed3

RATE = 44100


def record_audio():
    global recorded_data
    recorded_data = []

    def audio_callback(indata, frames, time, status):
        recorded_data.append(indata.copy())

    with sd.InputStream(samplerate=RATE, channels=1, callback=audio_callback):
        print("Recording started... Press Enter to stop recording.")
        input()
        print("Recording stopped.")


def callback(indata, frames, time, status):
    recorded_data.append(indata.copy())


def parse_timestamp(timestamp_str):
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return timestamp


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


if __name__ == "__main__":
    recorded_data = []
    sampling_rate = 44100
    num_channels = 1

    start_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    record_audio()
    print("\nRecording stopped.")
    end_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    save_audio_with_timestamps(start_timestamp, end_timestamp, recorded_data)
    print("Audio saved with start and end timestamps.")

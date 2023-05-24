#!/usr/bin/env python3
import argparse
from tinytag import TinyTag
import datetime


def read_timestamps_from_metadata(filename):
    audio = TinyTag.get(filename)

    for i in dir(audio):
        print(f"{i}: {getattr(audio, i)}")
    start_timestamp_str = audio.comment.split("\n")[0]
    end_timestamp_str = audio.comment.split("\n")[1]

    start_timestamp = datetime.datetime.fromisoformat(start_timestamp_str)
    end_timestamp = datetime.datetime.fromisoformat(end_timestamp_str)

    return start_timestamp, end_timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Read timestamps from metadata of a WAV file."
    )
    parser.add_argument("filename", help="Path to the WAV file")
    args = parser.parse_args()

    start_timestamp, end_timestamp = read_timestamps_from_metadata(args.filename)
    print(f"Start timestamp: {start_timestamp}")
    print(f"End timestamp: {end_timestamp}")


if __name__ == "__main__":
    main()

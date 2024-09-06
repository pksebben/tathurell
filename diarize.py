#!/usr/bin/env python3

import argparse
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment
from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import combined_waveplot

def load_sound_file(file_path):
    """
    Loads a sound file. Converts to WAV if necessary.
    """
    if file_path.endswith('.mp3'):
        return convert_mp3_to_wav(file_path)
    else:
        return file_path  # No conversion needed for WAV files

def convert_mp3_to_wav(mp3_file_path):
    """
    Converts an MP3 file to WAV and returns the path of the WAV file.
    """
    sound = AudioSegment.from_mp3(mp3_file_path)
    wav_file_path = mp3_file_path.replace('.mp3', '.wav')
    sound.export(wav_file_path, format="wav")
    return wav_file_path

def save_segments_to_csv(segments, csv_file_path):
    """
    Saves the segments to a CSV file.
    """
    pd.DataFrame(segments).to_csv(csv_file_path, index=False)

def main(wav_file_path, num_speakers, csv_file_path='segments.csv'):
    """
    Main function to handle diarization and output.
    """
    wav_file = load_sound_file(wav_file_path)

    diar = Diarizer(
                      embed_model='xvec',  # 'xvec' and 'ecapa' supported
                      cluster_method='sc'  # 'ahc' and 'sc' supported
                   )

    segments = diar.diarize(wav_file, num_speakers=num_speakers)

    save_segments_to_csv(segments, csv_file_path)
    print(f"Segmentation completed. Results saved to {csv_file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Diarization Script")
    parser.add_argument('file_path', type=str, help="Path to the audio file (WAV or MP3)")
    parser.add_argument('num_speakers', type=int, help="Number of speakers to identify in the audio file")
    parser.add_argument('--csv_file_path', type=str, default='segments.csv', help="Path to save the segments CSV file")

    args = parser.parse_args()

    main(args.file_path, args.num_speakers, args.csv_file_path)

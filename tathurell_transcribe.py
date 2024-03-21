#!/usr/bin/env python3
import sys
from vosk import Model, KaldiRecognizer
import json
from pydub import AudioSegment
import io
import numpy as np

audio_path = sys.argv[1]

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
)

# send pipeline to GPU (when available)
import torch
import torchaudio

pipeline.to(torch.device("mps"))

# apply pretrained pipeline
waveform, sample_rate = torchaudio.load(audio_path)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# print the result
turnlist = []

for turn, _, speaker in diarization.itertracks(yield_label=True):

    turnlist.append({"speaker": speaker, "start": turn.start, "end": turn.end})
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


# Load the model
model = Model("/Users/benmorsillo/code/JOAN/models/vosk-model-en-us-0.42-gigaspeech")

# Create a new recognizer with partial results config
rec = KaldiRecognizer(model, 16000, '{"partial_results": True}')

rec.SetWords(True)

# Convert the mp3 file to wav
audio = AudioSegment.from_mp3(audio_path)
audio = audio.set_frame_rate(16000).set_channels(1)
buf = io.BytesIO()
audio.export(buf, format="wav")
buf.seek(0)

wav_data = np.frombuffer(buf.read(), dtype=np.int16)

# Chunk size in seconds
chunk_duration = 4000 / 16000

# Total duration of all processed chunks
total_duration = 0

combined_transcription = []

# Process chunks of the audio data
print("PROCESSING")
for i in range(0, len(wav_data), 4000):
    chunk = wav_data[i : i + 4000]
    if rec.AcceptWaveform(chunk.tobytes()):
        res = json.loads(rec.Result())
        print(f"RES: {res}")
        # Adjust each word's timestamps
        if "result" in res:
            for word in res.get("result", []):
                for turn in turnlist:
                    if word["start"] > turn["end"]:
                        pass
                    else:
                        speaker = turn["speaker"]
                        break
                combined_transcription.append(
                    {"word": word["word"], "speaker": speaker}
                )
                word["start"] += total_duration
                word["end"] += total_duration
            print(res)
    # Update total_duration
    total_duration += chunk_duration

# Get the final result without timestamps
res = json.loads(rec.FinalResult())

current_speaker = None
current_chunk = ""
diarized_transcription = []

for word in combined_transcription:
    if word["speaker"] != current_speaker:
        if current_speaker is not None:
            diarized_transcription.append(
                {"speaker": current_speaker, "text": current_chunk}
            )
        current_chunk = ""
        current_speaker = word["speaker"]
    else:
        current_chunk += f" {word['word']}"

diarized_transcription.append({"speaker": current_speaker, "text": current_chunk})

names = {}

for chunk in diarized_transcription:
    if chunk["speaker"] in names.keys():
        pass
    else:
        print(f"Who said this?\n{chunk['text']}")
        name = input("input name:")
        names[chunk["speaker"]] = name

compiled_transcription = [
    f"{names[chunk['speaker']]}: {chunk['text']}" for chunk in diarized_transcription
]

with open(f"{audio_path}.transcription", "w") as f:
    f.write("\n".join(compiled_transcription))

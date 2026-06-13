#!/usr/bin/env python3
import os
import sys
from vosk import Model, KaldiRecognizer
import json
from pydub import AudioSegment
import io
import numpy as np

audio_path = sys.argv[1]

from pyannote.audio import Pipeline

# pyannote's gated models require a Hugging Face token. Keep it out of source:
# export HF_TOKEN=hf_... before running. (The previously hardcoded token was
# leaked in git history and must be revoked on huggingface.co.)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    sys.exit("HF_TOKEN environment variable is not set (needed for pyannote).")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token,
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
model = Model("/Users/benmorsillo/code/ASSISTANTS/JOAN/models/vosk-model-en-us-0.42-gigaspeech")

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

combined_transcription = []

# Process chunks of the audio data
print("PROCESSING")
for i in range(0, len(wav_data), 4000):
    chunk = wav_data[i : i + 4000]
    if rec.AcceptWaveform(chunk.tobytes()):
        res = json.loads(rec.Result())
        print(f"RES: {res}")
        if "result" in res:
            for word in res.get("result", []):
                # vosk word timestamps are absolute (relative to the start of the
                # whole stream — verified empirically, they do not reset per
                # utterance), so they share a reference frame with pyannote's
                # diarization turns. Attribute the word to the first turn that
                # hasn't ended before the word starts. If the word falls past the
                # last detected turn, fall back to that last speaker (otherwise
                # `speaker` would carry over from the previous word, or be unbound
                # on the very first word).
                speaker = turnlist[-1]["speaker"] if turnlist else None
                for turn in turnlist:
                    if word["start"] <= turn["end"]:
                        speaker = turn["speaker"]
                        break
                combined_transcription.append(
                    {"word": word["word"], "speaker": speaker}
                )

# Get the final result without timestamps
res = json.loads(rec.FinalResult())

current_speaker = None
current_chunk = ""
diarized_transcription = []

for word in combined_transcription:
    if word["speaker"] != current_speaker:
        # Speaker changed: flush the previous run, then start the new run with
        # the word that triggered the change. (Previously this reset the chunk
        # to "" without keeping the word, dropping the first word of every run.)
        if current_speaker is not None:
            diarized_transcription.append(
                {"speaker": current_speaker, "text": current_chunk}
            )
        current_speaker = word["speaker"]
        current_chunk = word["word"]
    else:
        current_chunk += f" {word['word']}"

# Flush the final run (skip if there was no transcription at all).
if current_speaker is not None:
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

with open(f"{audio_path}.transcription.txt", "w") as f:
    f.write("\n".join(compiled_transcription))

"""Baseline engine: vosk (gigaspeech) ASR + pyannote 3.1 diarization +
max-overlap attribution. Mirrors Tathurell's production pipeline, fixed:
includes the trailing FinalResult() (production drops it) so ASR coverage is
fair, and uses max-overlap attribution.
"""
import json
import os

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from pyannote.audio import Pipeline

from eval.engines.base import assign_speakers_max_overlap

VOSK_MODEL = "/Users/benmorsillo/code/ASSISTANTS/JOAN/models/vosk-model-en-us-0.42-gigaspeech"


class VoskPyannote:
    name = "vosk_pyannote"

    def __init__(self):
        self._model = Model(VOSK_MODEL)
        # pyannote >= 3.3 renamed use_auth_token -> token
        self._dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"]
        ).to(torch.device("mps"))

    def _diarize(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        out = self._dia({"waveform": waveform, "sample_rate": sr})
        # pyannote 4.x returns a DiarizeOutput whose Annotation is
        # .speaker_diarization; pyannote 3.x returned the Annotation directly.
        annotation = getattr(out, "speaker_diarization", out)
        return [
            {"speaker": spk, "start": turn.start, "end": turn.end}
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]

    def _asr(self, audio_path):
        rec = KaldiRecognizer(self._model, 16000)
        rec.SetWords(True)
        # set_sample_width(2) forces 16-bit PCM. The corpus loaders emit float32
        # WAVs; casting those samples straight to int16 truncates every value in
        # [-1, 1] to 0 (silence), which makes vosk error or emit garbage. pydub's
        # set_sample_width(2) does the proper float->int16 rescaling first.
        audio = (
            AudioSegment.from_file(audio_path)
            .set_frame_rate(16000)
            .set_channels(1)
            .set_sample_width(2)
        )
        # get_array_of_samples() returns the raw PCM samples without any header,
        # unlike np.frombuffer(wav_with_header) which would include the 44-byte WAV header.
        pcm = np.array(audio.get_array_of_samples(), dtype=np.int16)
        words = []
        for i in range(0, len(pcm), 4000):
            if rec.AcceptWaveform(pcm[i:i + 4000].tobytes()):
                for w in json.loads(rec.Result()).get("result", []):
                    words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        # vosk's streaming loop drops the last partial utterance unless FinalResult() is called.
        # Word timestamps from this loop are ABSOLUTE (relative to stream start, not per-utterance),
        # so they align directly with pyannote's absolute turn times — no offset needed.
        for w in json.loads(rec.FinalResult()).get("result", []):
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        return words

    def transcribe(self, audio_path):
        turns = self._diarize(audio_path)
        words = self._asr(audio_path)
        return assign_speakers_max_overlap(words, turns)

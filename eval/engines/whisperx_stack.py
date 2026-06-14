"""Bake-off adapter: thin wrapper over the production WhisperXTranscriber so the
bake-off and the production tool share ONE WhisperX implementation (DRY).

Importing this module must NOT require a token (WhisperXTranscriber resolves the
token only at construction, not import).
"""
from tathurell.whisperx_core import WhisperXTranscriber


class WhisperXStack:
    name = "whisperx"

    def __init__(self):
        self._t = WhisperXTranscriber()

    def transcribe(self, audio_path):
        return self._t.transcribe(audio_path)

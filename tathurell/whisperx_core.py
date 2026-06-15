"""WhisperX transcription engine, owned by the production tool.

Promoted from the bake-off adapter. Version facts (whisperx 3.8.6 + pyannote 4.0.4),
all verified during the bake-off and required for correctness:
  - DiarizationPipeline lives in whisperx.diarize (not top-level whisperx).
  - pyannote 4.x auth arg is `token=` (not `use_auth_token=`).
  - faster-whisper (ctranslate2) has no MPS backend -> device must be "cpu".
  - aligned word dicts have keys word/start/end (+ score); start/end absent on
    tokens alignment could not pin -> skip those.
"""
import os
from pathlib import Path

import whisperx

try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

from tathurell.ffmpeg import ensure_ffmpeg_on_path
from tathurell.realign import realign_speakers

# Convenience fallback, resolved against the CURRENT WORKING DIRECTORY — so it
# only finds the file when the tool is run from the repo root. $HF_TOKEN works
# from anywhere and is the primary source.
DEFAULT_TOKEN_FILE = "eval/data/.hf_token"


def resolve_hf_token(token_file: str = DEFAULT_TOKEN_FILE) -> str:
    """HF token from $HF_TOKEN, else from token_file, else exit with guidance."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path(token_file)
    if p.exists():
        tok = p.read_text().strip()
        if tok:
            return tok
    raise SystemExit(
        "HF_TOKEN is not set. Create a token at https://huggingface.co/settings/tokens, "
        "accept the gates for pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0, "
        "and pyannote/speaker-diarization-community-1, then either export HF_TOKEN=... or "
        f"write it to {token_file}."
    )


class WhisperXTranscriber:
    """Load WhisperX (large-v3, CPU) + pyannote diarization once; transcribe to words."""

    def __init__(self, model="large-v3", device="cpu", compute_type="int8", token=None):
        self._device = device
        self._model = whisperx.load_model(model, device, compute_type=compute_type)
        self._diarize = DiarizationPipeline(
            token=token if token is not None else resolve_hf_token(), device=device
        )

    def transcribe(self, audio_path: str) -> list:
        """Return [{"word", "start", "end", "speaker"}] for the audio file."""
        ensure_ffmpeg_on_path()  # bundled ffmpeg shadows any system one for load_audio
        audio = whisperx.load_audio(audio_path)
        result = self._model.transcribe(audio, batch_size=8)
        align_model, meta = whisperx.load_align_model(
            language_code=result["language"], device=self._device
        )
        result = whisperx.align(result["segments"], align_model, meta, audio, self._device)
        diar = self._diarize(audio)
        # fill_nearest=True so words in a diarization gap get the nearest speaker
        # instead of None (whisperx default leaves them unassigned).
        result = whisperx.assign_word_speakers(diar, result, fill_nearest=True)
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                if "start" not in w:  # alignment dropped timing for this token
                    continue
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": w.get("speaker"),
                })
        # whisperx assigns each word independently, so a single word at a turn
        # boundary can flip speaker mid-sentence. Realign per sentence by majority.
        return realign_speakers(words)

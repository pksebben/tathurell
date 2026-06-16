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

# Run fully offline from the local Hugging Face cache. The gated pyannote and
# whisper models are already downloaded, so no live HF token or network call is
# needed. setdefault lets a user opt back online (HF_HUB_OFFLINE=0) to fetch new
# models; this must run before whisperx/huggingface_hub is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import whisperx

try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

from tathurell.confidence import word_confidences
from tathurell.ffmpeg import ensure_ffmpeg_on_path
from tathurell.realign import realign_speakers


class WhisperXTranscriber:
    """Load WhisperX (large-v3, CPU) + pyannote diarization once; transcribe to words."""

    def __init__(self, model="large-v3", device="cpu", compute_type="int8"):
        self._device = device
        self._model = whisperx.load_model(model, device, compute_type=compute_type)
        # token=None: the gated models load from the local cache (offline mode
        # is set at import), so no HF token is required.
        self._diarize = DiarizationPipeline(token=None, device=device)

    def transcribe(self, audio_path: str, progress=None) -> list:
        """Return [{"word", "start", "end", "speaker"}] for the audio file.

        progress: optional callback(stage_name) invoked at each coarse pipeline
        stage ("transcribing"/"aligning"/"diarizing"/"finishing"). Default None
        (the CLI passes nothing -> unchanged behavior).
        """
        def _p(stage):
            if progress is not None:
                progress(stage)

        ensure_ffmpeg_on_path()  # bundled ffmpeg shadows any system one for load_audio
        audio = whisperx.load_audio(audio_path)
        _p("transcribing")
        result = self._model.transcribe(audio, batch_size=8)
        _p("aligning")
        align_model, meta = whisperx.load_align_model(
            language_code=result["language"], device=self._device
        )
        result = whisperx.align(result["segments"], align_model, meta, audio, self._device)
        _p("diarizing")
        diar = self._diarize(audio)
        _p("finishing")
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
        words = realign_speakers(words)
        # Attach per-word diarization confidence (overlap dominance of the final
        # speaker) so the UI can flag uncertain runs.
        diar_segments = list(zip(diar["start"], diar["end"], diar["speaker"]))
        for w, c in zip(words, word_confidences(words, diar_segments)):
            w["confidence"] = c
        return words

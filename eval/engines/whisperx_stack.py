"""WhisperX engine: faster-whisper (large-v3) + wav2vec2 forced alignment +
pyannote diarization + built-in word->speaker assignment. CPU on Apple Silicon
(ctranslate2 has no MPS backend) — that runtime cost is part of the eval.

Introspected against whisperx 3.8.6 + pyannote.audio 4.0.4:
  - DiarizationPipeline lives in whisperx.diarize (not exported at top-level
    whisperx namespace), so it must be imported from whisperx.diarize directly.
  - DiarizationPipeline.__init__ accepts token= (not use_auth_token=); pyannote
    4.x renamed the arg and whisperx passes it through to Pipeline.from_pretrained.
  - pyannote 4.x Pipeline.__call__ returns a DiarizeOutput object (not a bare
    Annotation) when self.legacy is False (the default).  whisperx.diarize
    accesses output.speaker_diarization which matches the 4.x non-legacy path.
  - Aligned word dicts from whisperx.align have keys: word, start, end, score.
    start/end may be absent when alignment cannot pin a token — the guard below
    skips those words.

Lazy-init contract: importing this module must NOT require HF_TOKEN. The
DiarizationPipeline (which calls out to HuggingFace) is constructed only inside
WhisperXStack.__init__, never at module scope.
"""
import os
import whisperx

# DiarizationPipeline is NOT exported from the top-level whisperx namespace.
# It lives in whisperx.diarize. The try/except below handles any future
# version that moves it back to the top level.
try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # CPU-friendly; bump to float32 if accuracy matters more than speed


class WhisperXStack:
    name = "whisperx"

    def __init__(self):
        self._model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
        # token= is the correct param name in whisperx 3.8.6 + pyannote 4.x;
        # use_auth_token= was the older pyannote 2.x name.
        self._diarize = DiarizationPipeline(token=os.environ["HF_TOKEN"], device=DEVICE)

    def transcribe(self, audio_path: str) -> list:
        """Transcribe audio_path and return a list of Word dicts.

        Each Word is {"word": str, "start": float, "end": float, "speaker": str|None}.
        Words without alignment timestamps (align drops timing on some tokens) are
        silently omitted rather than forwarded with missing keys.
        """
        audio = whisperx.load_audio(audio_path)

        # Step 1: ASR
        result = self._model.transcribe(audio, batch_size=8)

        # Step 2: Force-align to get per-word timestamps
        align_model, meta = whisperx.load_align_model(
            language_code=result["language"], device=DEVICE
        )
        result = whisperx.align(result["segments"], align_model, meta, audio, DEVICE)

        # Step 3: Diarize and assign each word to a speaker
        diar = self._diarize(audio)
        result = whisperx.assign_word_speakers(diar, result)

        # Step 4: Flatten segments -> word list, enforcing the Word contract
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                # Alignment can omit start/end for unresolvable tokens; skip them.
                if "start" not in w:
                    continue
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": w.get("speaker"),
                })
        return words

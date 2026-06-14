# Engine Bake-off — Results & Recommendation

**Run date:** 2026-06-14 · **Spec:** `2026-06-13-engine-bakeoff-design.md` · **Plan:** `../plans/2026-06-13-engine-bakeoff.md`
**Env:** `tathurell-eval` venv · pyannote.audio 4.0.4 · whisper large-v3 · vosk gigaspeech · Apple Silicon (MPS for pyannote, CPU for WhisperX)

## Results

Headline metric is **cpWER** (speaker-attributed); WER is ASR-only; DER is diarization. Lower is better. `sec` = wall-clock for that engine on that slice. Slices: AMI ES2011a far-field 10 min, Earnings-21 call 4386541 full ~18 min, CHiME-6 S02 distant 10 min.

| corpus (condition) | engine | WER | cpWER | DER | sec |
|---|---|---|---|---|---|
| **AMI-SDM** (far-field meeting) | vosk | 0.623 | 0.885 | 0.820 | 115 |
| | **whisperx** | **0.386** | **0.562** | **0.568** | 707 |
| | mlx-whisper | 0.477 | 0.876 | 0.906 | 163 |
| **Earnings-21** (telephony, clean) | vosk | 0.140 | 0.141 | N/A | 185 |
| | **whisperx** | **0.061** | **0.079** | N/A | 1372 |
| | mlx-whisper | 0.302 | 0.313 | N/A | 250 |
| **CHiME-6** (noisy dinner, far-field) | vosk | 0.834 | 1.013 | 0.766 | 181 |
| | **whisperx** | **0.678** | **0.810** | **0.749** | 785 |
| | mlx-whisper | 0.713 | 0.974 | 0.738 | 139 |

DER is N/A for Earnings-21 (its reference has no per-segment timestamps). `mem_MB` was dropped — `ru_maxrss` is a process-cumulative peak, so the per-engine delta read ~0 and is not meaningful.

## Reading

- **WhisperX wins every accuracy metric on every corpus** — WER, cpWER, and DER — usually by a wide margin (Earnings cpWER 0.079 vs next-best 0.141; AMI cpWER 0.562 vs 0.876).
- **Speed is the trade.** WhisperX is CPU-only on Apple Silicon (ctranslate2 has no MPS backend) and ran **3–7× slower** than mlx/vosk. mlx-whisper and vosk are comparably fast.
- **Speaker attribution is where mlx/vosk lose most.** Their cpWER inflates far above their WER (mlx AMI: WER 0.477 → cpWER 0.876), because they use separate pyannote + max-overlap attribution, whereas WhisperX assigns speakers to words integrally. WhisperX's cpWER stays much closer to its WER.
- **Absolute quality, not just ranking:** on clean telephony (Earnings) WhisperX is genuinely excellent (cpWER 0.079). On far-field multi-speaker audio (AMI/CHiME) **even the winner is mediocre** (cpWER 0.56–0.81). On that kind of audio, recording/mic quality dominates engine choice.

## Recommendation — adopt **WhisperX**

For Tathurell's stated **accuracy-first** goal, WhisperX is the clear pick: it wins on every corpus and every metric, including the clean telephony condition where the comparison is least confounded. Rebuild the transcription pipeline on WhisperX (Whisper large-v3 + forced alignment + integrated pyannote diarization + word-level speaker assignment), replacing the vosk+pyannote+max-overlap path.

### Caveats and seams (documented, not blockers)

1. **Runtime.** WhisperX is CPU-only on this hardware → a multi-hour recording takes hours to process. For an offline personal tool this is acceptable, but if it becomes painful, **mlx-whisper is the Apple-native fast path** (~5× faster), not a dead end — see below.
2. **mlx's gap is partly tunable.** mlx ran with default decoding (no VAD, default beam) against WhisperX's VAD + tuned/batched harness — same base model (large-v3), yet a 5× WER gap on Earnings. Much of mlx's deficit is likely harness/settings + the weaker separate-attribution step, not the model. If speed matters later, tuning mlx (VAD, decoding params, better word→speaker assignment) is the optimization path.
3. **N=1 per corpus.** One slice each (10–18 min). The ranking is consistent across all nine cells, but absolute numbers carry single-sample noise.
4. **Not a pure model comparison.** The bake-off compares *engine stacks as configured*, which is the right question for a deployment decision, but means "WhisperX ASR > mlx ASR" is partly a harness result.

### Why not the others
- **vosk (current baseline):** worst on the hard far-field corpora (CHiME cpWER 1.013); decent only on clean telephony. Its virtues are speed and being fully local/offline (no HF gating, no torch/MLX) — relevant only if those become hard requirements, which the accuracy-first goal does not make them.
- **mlx-whisper:** fast and Apple-native, but consistently 2nd–3rd on accuracy as configured. Reserve as the speed-optimization path, not the accuracy pick.

## Reproduce

```
pyenv activate tathurell-eval
export HF_TOKEN=<token w/ pyannote/speaker-diarization-3.1, segmentation-3.0, AND speaker-diarization-community-1 accepted>
python -m eval.run_bakeoff
```
Saved hypothesis transcripts are under `eval/data/<uri>.<engine>.txt`.

### Runtime fixes applied during this run (committed)
- pyannote 4.x returns `DiarizeOutput`, not `Annotation` → use `.speaker_diarization`.
- PyTorch + Apple MLX in one process segfaults → mlx ASR runs in a torch-free subprocess (`eval/engines/_mlx_worker.py`).
- vosk was fed float32 WAVs cast to int16 (→ silence) → `set_sample_width(2)` rescales first. (vosk rows above were regenerated on the same slice WAVs after this fix; whisperx/mlx rows are from the main run.)

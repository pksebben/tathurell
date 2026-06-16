#!/usr/bin/env python3
"""Transcribe + diarize an audio file with WhisperX, name speakers, write text.

Output: "<name>: <text>" lines, one per speaker run, to <audio>.transcription.txt
(or --output). Runs in the tathurell-eval venv. Needs HF_TOKEN (see whisperx_core).
"""
import argparse
import sys

from tathurell.naming import apply_names, group_by_speaker
from tathurell.whisperx_core import WhisperXTranscriber


def prompt_names(groups):
    """Ask the user to name each distinct speaker (once). EOF -> use the label."""
    names = {}
    for g in groups:
        spk = g["speaker"]
        if spk in names:
            continue
        print(f"Who said this?\n{g['text']}")
        try:
            answer = input("name: ").strip()
        except EOFError:
            answer = ""
        names[spk] = answer or str(spk)
    return names


def resolve_names(words, groups, audio_path, no_ui):
    """Get {speaker: name}: browser modal by default; terminal prompts on --no-ui
    or if the modal can't run (headless/no browser)."""
    if no_ui:
        return prompt_names(groups)
    try:
        import tathurell.naming_ui as naming_ui
        from tathurell.sampling import pick_speaker_samples
        return naming_ui.collect_names(pick_speaker_samples(words), audio_path)
    except Exception as exc:
        print(f"[tathurell] naming modal unavailable ({exc}); using terminal prompts",
              file=sys.stderr)
        return prompt_names(groups)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Transcribe + diarize audio with WhisperX.")
    ap.add_argument("audio_path")
    ap.add_argument("--output", default=None,
                    help="output path (default: <audio_path>.transcription.txt)")
    ap.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3)")
    ap.add_argument("--no-ui", action="store_true",
                    help="skip the browser naming modal; name speakers via terminal prompts")
    args = ap.parse_args(argv)

    words = WhisperXTranscriber(model=args.model).transcribe(args.audio_path)
    if not words:
        print("[tathurell] WARNING: no words transcribed.", file=sys.stderr)

    groups = group_by_speaker(words)
    names = resolve_names(words, groups, args.audio_path, args.no_ui)
    text = apply_names(groups, names)

    out_path = args.output or f"{args.audio_path}.transcription.txt"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[tathurell] wrote {out_path}")


if __name__ == "__main__":
    main()

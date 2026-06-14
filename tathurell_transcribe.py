#!/usr/bin/env python3
"""Transcribe + diarize an audio file with WhisperX, name speakers, write text.

Output: "<name>: <text>" lines, one per speaker run, to <audio>.transcription.txt
(or --output). Runs in the tathurell-eval venv. Needs HF_TOKEN (see whisperx_core).
"""
import argparse
import sys

from tathurell.naming import apply_names, group_by_speaker
from tathurell.whisperx_core import WhisperXTranscriber, resolve_hf_token


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Transcribe + diarize audio with WhisperX.")
    ap.add_argument("audio_path")
    ap.add_argument("--output", default=None,
                    help="output path (default: <audio_path>.transcription.txt)")
    ap.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3)")
    args = ap.parse_args(argv)

    token = resolve_hf_token()  # clean early exit before loading models if missing
    words = WhisperXTranscriber(model=args.model, token=token).transcribe(args.audio_path)
    if not words:
        print("[tathurell] WARNING: no words transcribed.", file=sys.stderr)

    groups = group_by_speaker(words)
    names = prompt_names(groups)
    text = apply_names(groups, names)

    out_path = args.output or f"{args.audio_path}.transcription.txt"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[tathurell] wrote {out_path}")


if __name__ == "__main__":
    main()

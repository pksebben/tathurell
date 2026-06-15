"""Front-door transcription web app (Phase 1).

upload -> transcribe (background job, coarse stage progress) -> name speakers
-> inline preview + download. Persistent, one-shot, single job at a time.
Launched via `python -m tathurell.webapp`. The CLI (tathurell_transcribe.py) and
naming_ui.collect_names are unaffected.
"""
import os
import shutil
import sys
import tempfile
import threading
import webbrowser

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import make_server

from tathurell.naming import apply_names, group_by_speaker
from tathurell.sampling import extract_clip, pick_speaker_samples
from tathurell.whisperx_core import WhisperXTranscriber

# Stages reported while the background job runs (before naming).
_RUNNING_STAGES = ("transcribing", "aligning", "diarizing", "finishing")


class Job:
    """Single in-process transcription job: one-shot, lock-guarded state.

    Lifecycle of `stage`: idle -> transcribing/aligning/diarizing/finishing
    -> naming -> done (or -> error at any point). reset() returns to idle and
    removes the job's temp dir.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._tmpdir = None
        self._init_state()

    def _init_state(self):
        self.stage = "idle"
        self.error = None
        self.audio_name = None
        self.samples = None  # {speaker: {"start","end","text"}}
        self.groups = None   # [{"speaker","text"}]
        self.text = None     # final named transcript

    @property
    def tmpdir(self):
        return self._tmpdir

    def start(self, tmpdir, audio_name):
        with self.lock:
            self._tmpdir = tmpdir
            self.audio_name = audio_name
            self.error = None
            self.stage = "transcribing"

    def set_stage(self, stage):
        with self.lock:
            self.stage = stage

    def set_naming(self, groups, samples):
        with self.lock:
            self.groups = groups
            self.samples = samples
            self.stage = "naming"

    def set_done(self, text):
        with self.lock:
            self.text = text
            self.stage = "done"

    def set_error(self, message):
        with self.lock:
            self.stage = "error"
            self.error = message

    def reset(self):
        with self.lock:
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
                self._tmpdir = None
            self._init_state()

    def snapshot(self):
        """JSON-safe view for GET /status."""
        with self.lock:
            snap = {"stage": self.stage}
            if self.error:
                snap["error"] = self.error
            if self.stage == "naming" and self.samples:
                snap["speakers"] = [
                    {"id": spk, "text": s["text"]} for spk, s in self.samples.items()
                ]
            return snap

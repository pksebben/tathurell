"""Diarization Error Rate via pyannote.metrics. Inputs are RTTM file paths.
Diagnostic only (pyannote is shared across most stacks), but useful to confirm
diarization isn't the differentiator.
"""
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm


def _single_annotation(rttm_path):
    annotations = load_rttm(rttm_path)  # {uri: Annotation}
    return next(iter(annotations.values()))


def diarization_error_rate(reference_rttm: str, hypothesis_rttm: str) -> float:
    metric = DiarizationErrorRate()
    return float(metric(_single_annotation(reference_rttm), _single_annotation(hypothesis_rttm)))

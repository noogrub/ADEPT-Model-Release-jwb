"""Empirical tail calibration for scalar prediction-error scores."""

from bisect import bisect_left
from math import isfinite, log2


def prepare_reference(scores):
    """Return validated reference scores in ascending order."""
    reference = tuple(sorted(float(score) for score in scores))

    if not reference:
        raise ValueError("reference scores cannot be empty")

    if not all(isfinite(score) for score in reference):
        raise ValueError("reference scores must be finite")

    return reference


def calibrate_value(score, reference):
    """Return the tail count, tail probability, and SVoE bits."""
    score = float(score)

    if not isfinite(score):
        raise ValueError("score must be finite")

    n_reference = len(reference)
    first_equal_or_larger = bisect_left(reference, score)
    tail_count = n_reference - first_equal_or_larger

    tail_probability = (tail_count + 1) / (n_reference + 1)
    svoe_bits = -log2(tail_probability)

    return tail_count, tail_probability, svoe_bits


def calibrate_score(score, reference):
    """Convert one score to a complete calibration record."""
    score = float(score)

    tail_count, tail_probability, svoe_bits = calibrate_value(
        score,
        reference,
    )

    return {
        "score": score,
        "tail_count": tail_count,
        "tail_probability": tail_probability,
        "svoe_bits": svoe_bits,
    }


def calibrate_scores(scores, reference_scores):
    """Calibrate a sequence of scores against ordinary reference scores."""
    reference = prepare_reference(reference_scores)
    return [calibrate_score(score, reference) for score in scores]

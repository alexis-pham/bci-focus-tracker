"""
preprocessor.py — Filtering, epoching, and artifact rejection.

Design notes:
  - Each session is filtered independently to avoid edge effects bleeding
    across session boundaries.
  - sosfiltfilt (zero-phase) is used for both filters to avoid phase
    distortion, which matters for the frequency band power estimates.
  - Artifact rejection is amplitude-based (peak-to-peak threshold per
    channel per epoch). Simple and effective for a person-specific model.
  - Epoch labels are assigned by majority vote within the window, which
    correctly handles the (rare) case of a window spanning two sessions
    in the concatenated DataFrame. In practice this shouldn't occur since
    we process sessions independently.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos
from typing import Tuple

from config import (
    SAMPLE_RATE,
    NOTCH_FREQ, NOTCH_Q,
    BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER,
    WINDOW_SAMPLES, STEP_SAMPLES,
    ARTIFACT_THRESHOLD_UV,
)

# ---------------------------------------------------------------------------
# Filter construction — built once, reused across all sessions
# ---------------------------------------------------------------------------

def _build_notch_sos(fs: float = SAMPLE_RATE) -> np.ndarray:
    """60 Hz notch filter as second-order sections."""
    b, a = iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=fs)
    return tf2sos(b, a)


def _build_bandpass_sos(fs: float = SAMPLE_RATE) -> np.ndarray:
    """1–50 Hz Butterworth bandpass as second-order sections."""
    nyq = fs / 2.0
    sos = butter(
        FILTER_ORDER,
        [BANDPASS_LOW / nyq, BANDPASS_HIGH / nyq],
        btype="band",
        output="sos",
    )
    return sos


# Pre-build at import time — same coefficients used for every session
_NOTCH_SOS    = _build_notch_sos()
_BANDPASS_SOS = _build_bandpass_sos()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_eeg(eeg: np.ndarray) -> np.ndarray:
    """
    Apply notch (60 Hz) then bandpass (1–50 Hz) to raw EEG.

    Args:
        eeg: (n_samples, n_channels) raw EEG in µV

    Returns:
        filtered: same shape, zero-phase filtered
    """
    # sosfiltfilt applies the filter forward and backward (zero phase distortion)
    filtered = sosfiltfilt(_NOTCH_SOS,    eeg, axis=0)
    filtered = sosfiltfilt(_BANDPASS_SOS, filtered, axis=0)
    return filtered


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------

def epoch_signal(
    eeg:         np.ndarray,
    labels:      np.ndarray,
    session_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice a continuous filtered EEG array into overlapping windows.

    Args:
        eeg:         (n_samples, n_channels)
        labels:      (n_samples,) integer class labels
        session_ids: (n_samples,) string session identifiers

    Returns:
        epochs:      (n_epochs, window_samples, n_channels)
        ep_labels:   (n_epochs,)
        ep_sessions: (n_epochs,)
    """
    n_samples = eeg.shape[0]
    starts = range(0, n_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES)

    epochs, ep_labels, ep_sessions = [], [], []

    for start in starts:
        end = start + WINDOW_SAMPLES
        window = eeg[start:end]

        # Majority-vote label for this window
        window_labels        = labels[start:end]
        unique, counts       = np.unique(window_labels, return_counts=True)
        majority_label       = unique[np.argmax(counts)]

        epochs.append(window)
        ep_labels.append(majority_label)
        ep_sessions.append(session_ids[start])   # all the same within a session

    return (
        np.array(epochs,      dtype=np.float64),
        np.array(ep_labels,   dtype=np.int64),
        np.array(ep_sessions, dtype=object),
    )


# ---------------------------------------------------------------------------
# Artifact rejection
# ---------------------------------------------------------------------------

def reject_artifacts(
    epochs:      np.ndarray,
    labels:      np.ndarray,
    session_ids: np.ndarray,
    session_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop epochs where any channel exceeds the peak-to-peak amplitude threshold.

    A peak-to-peak value above 100 µV almost certainly reflects an eye blink,
    jaw clench, or movement artifact rather than neural signal.

    Args:
        epochs:      (n_epochs, window_samples, n_channels)
        labels:      (n_epochs,)
        session_ids: (n_epochs,)
        session_name: used only for logging

    Returns:
        Filtered versions of all three arrays.
    """
    # peak-to-peak per epoch per channel → (n_epochs, n_channels)
    ptp   = epochs.max(axis=1) - epochs.min(axis=1)
    clean = ptp.max(axis=1) < ARTIFACT_THRESHOLD_UV   # (n_epochs,) boolean

    n_total    = len(epochs)
    n_rejected = (~clean).sum()
    pct        = 100 * n_rejected / n_total if n_total > 0 else 0

    label = f"[{session_name}] " if session_name else ""
    print(
        f"    {label}Artifact rejection: "
        f"kept {clean.sum()}/{n_total} epochs "
        f"(removed {n_rejected}, {pct:.1f}%)"
    )

    return epochs[clean], labels[clean], session_ids[clean]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full preprocessing pipeline on a loaded DataFrame.

    Each session is processed independently (filter → epoch → reject)
    to prevent filter edge effects from contaminating session boundaries.

    Args:
        df: output of loader.load_all_sessions()

    Returns:
        all_epochs:   (n_epochs, window_samples, n_channels)
        all_labels:   (n_epochs,)
        all_sessions: (n_epochs,) session_id strings
    """
    channel_cols = [f"ch{i}" for i in range(1, 9)]

    all_epochs, all_labels, all_sessions = [], [], []

    for session_id in sorted(df["session_id"].unique()):
        sess_df = df[df["session_id"] == session_id].reset_index(drop=True)

        eeg         = sess_df[channel_cols].values    # (n_samples, 8)
        labels      = sess_df["label"].values
        session_ids = sess_df["session_id"].values

        filtered               = filter_eeg(eeg)
        epochs, ep_lab, ep_ses = epoch_signal(filtered, labels, session_ids)
        epochs, ep_lab, ep_ses = reject_artifacts(epochs, ep_lab, ep_ses, session_id)

        print(f"    [{session_id}] → {len(epochs)} clean epochs")

        all_epochs.append(epochs)
        all_labels.append(ep_lab)
        all_sessions.append(ep_ses)

    epochs_out   = np.concatenate(all_epochs,   axis=0)
    labels_out   = np.concatenate(all_labels,   axis=0)
    sessions_out = np.concatenate(all_sessions, axis=0)

    n_focused    = (labels_out == 0).sum()
    n_distracted = (labels_out == 1).sum()
    print(
        f"\n  Total clean epochs: {len(epochs_out)} "
        f"(focused: {n_focused}, distracted: {n_distracted})"
    )

    return epochs_out, labels_out, sessions_out

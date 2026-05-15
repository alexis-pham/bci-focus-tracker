"""
config.py — Single source of truth for all pipeline constants.

This file is shared between the training pipeline and the live inference
pipeline. Never hardcode these values elsewhere — import from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = Path("data")
FOCUSED_DIR     = DATA_DIR / "focused"
DISTRACTED_DIR  = DATA_DIR / "distracted"
MODEL_DIR       = Path("model")
MODEL_PATH      = MODEL_DIR / "bci_classifier.pkl"

# ---------------------------------------------------------------------------
# Hardware / data format
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 250          # Hz — confirmed from timestamp deltas
N_CHANNELS      = 8            # EEG channels
EEG_COLS        = list(range(1, 9))   # CSV column indices for EEG (1–8)
TIMESTAMP_COL   = 22           # Unix timestamp column

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_FOCUSED    = 0
LABEL_DISTRACTED = 1
LABEL_NAMES      = {LABEL_FOCUSED: "focused", LABEL_DISTRACTED: "distracted"}

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
NOTCH_FREQ      = 60.0         # Hz — US power line noise
NOTCH_Q         = 30.0         # Quality factor for notch sharpness
BANDPASS_LOW    = 1.0          # Hz — removes DC drift
BANDPASS_HIGH   = 50.0         # Hz — removes high-freq muscle noise
FILTER_ORDER    = 4            # Butterworth order (higher = sharper roll-off)

# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------
WINDOW_SEC      = 2.0          # seconds per epoch
OVERLAP         = 0.5          # 50% overlap between consecutive windows
WINDOW_SAMPLES  = int(WINDOW_SEC * SAMPLE_RATE)   # 500 samples
STEP_SAMPLES    = int(WINDOW_SAMPLES * (1 - OVERLAP))  # 250 samples

# ---------------------------------------------------------------------------
# Artifact rejection
# ---------------------------------------------------------------------------
# OpenBCI/BrainFlow recordings have larger raw amplitudes than textbook EEG
# (often 200–2000 µV peak-to-peak depending on electrode contact quality).
# 500 µV retains ~95% of clean epochs while removing genuine transient artifacts.
# If you are seeing too many rejections, raise this value; if the classifier
# seems noisy, lower it. Run the pipeline once and read the rejection logs.
ARTIFACT_THRESHOLD_UV = 500.0  # µV — peak-to-peak per channel per epoch

# ---------------------------------------------------------------------------
# Frequency bands (Hz) — ordered: theta, alpha, beta
# ---------------------------------------------------------------------------
FREQ_BANDS = {
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
}

# ---------------------------------------------------------------------------
# PSD estimation (Welch)
# ---------------------------------------------------------------------------
# Using full window length gives frequency resolution of 1/WINDOW_SEC = 0.5 Hz
WELCH_NPERSEG = WINDOW_SAMPLES

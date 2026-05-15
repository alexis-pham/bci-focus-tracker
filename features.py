"""
features.py — Feature extraction from clean EEG epochs.

Feature vector per epoch (40 values total):
  For each of 8 channels:
    - Relative theta power   (theta / total)
    - Relative alpha power   (alpha / total)
    - Relative beta power    (beta  / total)
    - Theta / Beta ratio     (attention index)
    - Alpha / Beta ratio     (relaxation index)

Relative band power normalizes for individual differences in overall signal
amplitude. The ratios capture the two most validated attention indices in
the neurofeedback literature (high theta/beta and alpha/beta are associated
with distraction / reduced cognitive engagement).

Using Welch's method for PSD estimation: it averages multiple overlapping
sub-windows, making the power estimate more stable than a single FFT.
With WINDOW_SAMPLES=500 and SAMPLE_RATE=250, frequency resolution = 0.5 Hz.
"""

import numpy as np
from scipy.signal import welch

from config import (
    SAMPLE_RATE,
    FREQ_BANDS,
    WELCH_NPERSEG,
    N_CHANNELS,
)

# Feature dimensionality: 3 relative band powers + 2 ratios = 5 per channel
N_FEATURES_PER_CHANNEL = len(FREQ_BANDS) + 2   # 5
N_FEATURES_TOTAL       = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 40

# Human-readable feature names (useful for inspection / debugging)
FEATURE_NAMES = [
    f"ch{ch}_{feat}"
    for ch in range(1, N_CHANNELS + 1)
    for feat in list(FREQ_BANDS.keys()) + ["theta_beta_ratio", "alpha_beta_ratio"]
]


# ---------------------------------------------------------------------------
# Per-epoch feature extraction
# ---------------------------------------------------------------------------

def _band_power(psd: np.ndarray, freqs: np.ndarray) -> dict:
    """
    Integrate PSD within each frequency band.

    Args:
        psd:   power spectral density (n_freqs,)
        freqs: frequency axis (n_freqs,)

    Returns:
        dict mapping band name → absolute power (µV² / Hz integrated)
    """
    freq_res = freqs[1] - freqs[0]
    powers = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        mask           = (freqs >= low) & (freqs <= high)
        powers[band_name] = float(np.sum(psd[mask]) * freq_res)
    return powers


def extract_features(epoch: np.ndarray) -> np.ndarray:
    """
    Extract the 40-dimensional feature vector from one epoch.

    Args:
        epoch: (window_samples, n_channels) filtered EEG in µV

    Returns:
        features: (40,) float64 vector
    """
    features = np.empty(N_FEATURES_TOTAL, dtype=np.float64)
    idx = 0

    for ch in range(N_CHANNELS):
        freqs, psd = welch(
            epoch[:, ch],
            fs=SAMPLE_RATE,
            nperseg=WELCH_NPERSEG,
            window="hann",
            average="mean",
        )

        abs_power = _band_power(psd, freqs)
        theta = abs_power["theta"]
        alpha = abs_power["alpha"]
        beta  = abs_power["beta"]

        total = theta + alpha + beta
        safe_total = total if total > 0 else 1e-10
        safe_beta  = beta  if beta  > 0 else 1e-10

        # Relative band powers
        features[idx + 0] = theta / safe_total
        features[idx + 1] = alpha / safe_total
        features[idx + 2] = beta  / safe_total

        # Attention ratios
        features[idx + 3] = theta / safe_beta   # theta/beta
        features[idx + 4] = alpha / safe_beta   # alpha/beta

        idx += N_FEATURES_PER_CHANNEL

    return features


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def build_feature_matrix(epochs: np.ndarray) -> np.ndarray:
    """
    Extract features for all epochs.

    Args:
        epochs: (n_epochs, window_samples, n_channels)

    Returns:
        X: (n_epochs, 40) feature matrix
    """
    n_epochs = len(epochs)
    X = np.empty((n_epochs, N_FEATURES_TOTAL), dtype=np.float64)

    for i, epoch in enumerate(epochs):
        X[i] = extract_features(epoch)

        if (i + 1) % 500 == 0 or (i + 1) == n_epochs:
            print(f"    Features extracted: {i + 1}/{n_epochs}")

    print(f"  Feature matrix: {X.shape}  ({N_FEATURES_TOTAL} features/epoch)")
    return X

"""
loader.py — Reads BrainFlow CSVs from focused/ and distracted/ folders.

Each file is tab-separated with no header row. The relevant columns are:
  - 1–8:  EEG channels (µV)
  - 22:   Unix timestamp

Each loaded sample is tagged with its label (0=focused, 1=distracted)
and a unique session_id derived from the filename. The session_id is
what makes Leave-One-Session-Out cross-validation possible in trainer.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from config import (
    FOCUSED_DIR, DISTRACTED_DIR,
    EEG_COLS, TIMESTAMP_COL,
    LABEL_FOCUSED, LABEL_DISTRACTED,
    N_CHANNELS,
)


def load_session(filepath: Path, label: int) -> pd.DataFrame:
    """
    Load a single BrainFlow CSV file.

    Returns a DataFrame with columns:
        ch1..ch8   — EEG signal (µV)
        timestamp  — Unix timestamp (seconds)
        label      — class label (0 or 1)
        session_id — unique string identifying this recording
    """
    raw = pd.read_csv(filepath, sep="\t", header=None, dtype=np.float64)

    # Validate expected number of columns
    if raw.shape[1] < max(EEG_COLS) + 1 or raw.shape[1] <= TIMESTAMP_COL:
        raise ValueError(
            f"{filepath.name}: expected at least {TIMESTAMP_COL + 1} columns, "
            f"got {raw.shape[1]}"
        )

    df = pd.DataFrame()
    for i, col_idx in enumerate(EEG_COLS, start=1):
        df[f"ch{i}"] = raw[col_idx].values

    df["timestamp"]  = raw[TIMESTAMP_COL].values
    df["label"]      = label
    df["session_id"] = f"{_label_name(label)}_{filepath.stem}"

    return df


def load_all_sessions() -> pd.DataFrame:
    """
    Load all CSVs from both class folders and return a single DataFrame.

    Sessions from the same class are kept separate via session_id so that
    the cross-validator can hold out one session at a time.
    """
    frames: List[pd.DataFrame] = []

    for label, folder in [
        (LABEL_FOCUSED,    FOCUSED_DIR),
        (LABEL_DISTRACTED, DISTRACTED_DIR),
    ]:
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in '{folder}'. "
                "Make sure the folder exists and contains .csv files."
            )

        for filepath in csv_files:
            df = load_session(filepath, label)
            duration_sec = len(df) / 250  # approximate — exact rate in config
            print(
                f"  [{_label_name(label)}] {filepath.name}: "
                f"{len(df):,} samples  (~{duration_sec:.1f}s)"
            )
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Summary
    n_focused    = (combined["label"] == LABEL_FOCUSED).sum()
    n_distracted = (combined["label"] == LABEL_DISTRACTED).sum()
    n_sessions   = combined["session_id"].nunique()
    print(
        f"\n  Loaded {n_sessions} sessions | "
        f"{n_focused:,} focused samples | "
        f"{n_distracted:,} distracted samples"
    )

    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label_name(label: int) -> str:
    return "focused" if label == LABEL_FOCUSED else "distracted"

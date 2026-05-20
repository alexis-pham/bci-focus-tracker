"""
replay.py — Simulates the live BCI pipeline using a pre-recorded CSV file.

Replays a recording through the full preprocessing + feature extraction +
classification pipeline, window by window, at configurable speed. Serves
the current prediction over HTTP so the browser extension can poll it.
Generates a timeline visualization when the replay finishes.

Usage:
    python replay.py --file data/focused/BrainFlow-RAW_Recordings_10.csv
    python replay.py --file data/distracted/BrainFlow-RAW_Recordings_2.csv --speed 5

Arguments:
    --file   Path to the CSV recording to replay (required)
    --speed  Replay speed multiplier (default: 10 — 10x faster than real time)
    --port   Port for the local HTTP server (default: 5000)

The browser extension polls http://localhost:5000/status for predictions.
"""

import argparse
import threading
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

from config import (
    EEG_COLS, SAMPLE_RATE,
    WINDOW_SAMPLES, STEP_SAMPLES,
    ARTIFACT_THRESHOLD_UV,
    LABEL_FOCUSED, LABEL_DISTRACTED, LABEL_NAMES,
    MODEL_PATH,
)
from preprocessor import filter_eeg
from features import extract_features


# ---------------------------------------------------------------------------
# Shared state between replay thread and Flask server
# ---------------------------------------------------------------------------

_state = {
    "status":      "waiting",      # waiting | running | artifact | done
    "prediction":  "unknown",      # focused | distracted | artifact | unknown
    "label":       -1,             # 0=focused, 1=distracted, -1=artifact/unknown
    "window_idx":  0,
    "total_windows": 0,
    "elapsed_sec": 0.0,
    "progress_pct": 0.0,
}
_state_lock = threading.Lock()
_timeline   = []   # list of dicts — populated as replay runs


# ---------------------------------------------------------------------------
# Flask server
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)   # allow the extension (a different origin) to poll this server


@app.route("/status")
def status():
    """Current prediction — polled by the browser extension."""
    with _state_lock:
        return jsonify(dict(_state))


@app.route("/timeline")
def timeline():
    """Full prediction history — used to generate the plot."""
    return jsonify(_timeline)


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Replay thread
# ---------------------------------------------------------------------------

def _replay(csv_path: Path, speed: float) -> None:
    """
    Load a recording, filter it, then step through windows at the configured
    speed, classifying each one and updating shared state.
    """
    print(f"\n  Loading {csv_path.name}...")
    raw      = pd.read_csv(csv_path, sep="\t", header=None, dtype=np.float64)
    eeg      = raw[EEG_COLS].values
    n_raw    = len(eeg)
    duration = n_raw / SAMPLE_RATE
    print(f"  {n_raw:,} samples  (~{duration:.1f}s)")

    print("  Filtering (notch + bandpass)...")
    filtered = filter_eeg(eeg)

    # Load model
    model = joblib.load(MODEL_PATH)
    print(f"  Model loaded from {MODEL_PATH}")

    # Window indices
    starts       = list(range(0, len(filtered) - WINDOW_SAMPLES + 1, STEP_SAMPLES))
    n_windows    = len(starts)
    step_sleep   = (STEP_SAMPLES / SAMPLE_RATE) / speed   # seconds to sleep per step

    with _state_lock:
        _state["total_windows"] = n_windows
        _state["status"]        = "running"

    print(f"\n  Replaying {n_windows} windows at {speed}x speed")
    print(f"  Estimated demo duration: {n_windows * step_sleep:.1f}s")
    print(f"  Serving at http://localhost:{app.config.get('PORT', 5000)}/status\n")
    print(f"  {'Time':>8}  {'Window':>10}  Prediction")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}")

    for i, start in enumerate(starts):
        end    = start + WINDOW_SAMPLES
        window = filtered[start:end]

        elapsed     = start / SAMPLE_RATE
        progress    = (i + 1) / n_windows * 100

        # Artifact check
        ptp         = window.max(axis=0) - window.min(axis=0)
        is_artifact = bool(ptp.max() >= ARTIFACT_THRESHOLD_UV)

        if is_artifact:
            prediction = "artifact"
            label      = -1
        else:
            features   = extract_features(window).reshape(1, -1)
            label      = int(model.predict(features)[0])
            prediction = LABEL_NAMES.get(label, "artifact")

        # Update shared state
        with _state_lock:
            _state["status"]       = "artifact" if is_artifact else "running"
            _state["prediction"]   = prediction
            _state["label"]        = label
            _state["window_idx"]   = i
            _state["elapsed_sec"]  = elapsed
            _state["progress_pct"] = progress

        # Append to timeline
        _timeline.append({
            "time_sec":   elapsed,
            "prediction": prediction,
            "label":      label,
            "is_artifact": is_artifact,
        })

        # Terminal progress (every 20 windows)
        if i % 20 == 0 or i == n_windows - 1:
            symbol = "✓" if label == LABEL_FOCUSED else ("✗" if label == LABEL_DISTRACTED else "~")
            print(f"  {elapsed:>7.1f}s  {i+1:>4}/{n_windows:<4}  {symbol} {prediction}")

        time.sleep(step_sleep)

    # Replay finished
    with _state_lock:
        _state["status"]      = "done"
        _state["prediction"]  = "done"
        _state["progress_pct"] = 100.0

    print("\n  Replay complete. Generating timeline plot...")
    _save_timeline_plot(csv_path.stem)


# ---------------------------------------------------------------------------
# Timeline visualization
# ---------------------------------------------------------------------------

def _save_timeline_plot(session_name: str) -> None:
    """
    Save a timeline plot showing focused / distracted / artifact predictions
    over the course of the recording.
    """
    if not _timeline:
        return

    times       = np.array([e["time_sec"]   for e in _timeline])
    labels      = np.array([e["label"]       for e in _timeline])
    is_artifact = np.array([e["is_artifact"] for e in _timeline])

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # ── Top panel: prediction timeline ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#16213e")

    # Colour each window
    for i, (t, label, artifact) in enumerate(zip(times, labels, is_artifact)):
        width = STEP_SAMPLES / SAMPLE_RATE
        if artifact:
            color = "#555577"
        elif label == LABEL_FOCUSED:
            color = "#00d4aa"
        else:
            color = "#ff6b6b"
        ax.barh(0, width, left=t, height=0.8, color=color, alpha=0.85)

    # Legend
    patches = [
        mpatches.Patch(color="#00d4aa", label="Focused"),
        mpatches.Patch(color="#ff6b6b", label="Distracted"),
        mpatches.Patch(color="#555577", label="Artifact (skipped)"),
    ]
    ax.legend(handles=patches, loc="upper right", framealpha=0.3,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="none")

    ax.set_xlim(times[0], times[-1] + STEP_SAMPLES / SAMPLE_RATE)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", color="white")
    ax.set_title(f"Prediction Timeline — {session_name}", color="white", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    # ── Bottom panel: rolling distraction rate ────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")

    window_size = 20   # rolling window in epochs
    clean_mask  = ~is_artifact
    distract_rate = np.zeros(len(labels))
    for i in range(len(labels)):
        lo  = max(0, i - window_size)
        seg = labels[lo:i+1][clean_mask[lo:i+1]]
        distract_rate[i] = (seg == LABEL_DISTRACTED).mean() if len(seg) > 0 else 0

    ax2.fill_between(times, distract_rate, alpha=0.6, color="#ff6b6b")
    ax2.plot(times, distract_rate, color="#ff6b6b", linewidth=1)
    ax2.axhline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Time (seconds)", color="white")
    ax2.set_ylabel("Distraction\nRate", color="white", fontsize=8)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444466")

    plt.tight_layout(pad=1.5)

    out_path = Path(f"timeline_{session_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Timeline saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BCI replay pipeline")
    parser.add_argument(
        "--file", required=True, type=Path,
        help="Path to BrainFlow CSV recording to replay"
    )
    parser.add_argument(
        "--speed", type=float, default=10.0,
        help="Replay speed multiplier (default: 10 = 10x faster than real time)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port for the local HTTP server (default: 5000)"
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file not found: {args.file}")
        return

    app.config["PORT"] = args.port

    # Start replay in background thread
    t = threading.Thread(target=_replay, args=(args.file, args.speed), daemon=True)
    t.start()

    # Flask runs in main thread — blocks until Ctrl+C
    print(f"\n  BCI Replay Server starting on http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

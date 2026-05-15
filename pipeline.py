"""
pipeline.py — Main entry point for the BCI training pipeline.

Usage:
    python pipeline.py

Stages:
    1. Load    — reads all CSVs from data/focused/ and data/distracted/
    2. Preprocess — notch + bandpass filter, epoch, artifact rejection
    3. Features   — relative band power + attention ratios (40 features/epoch)
    4. Train      — LOSO cross-validation on SVM and LDA, save best model

Output:
    model/bci_classifier.pkl — trained scikit-learn Pipeline
                                (StandardScaler + best classifier)

    The saved pipeline is the only artifact consumed by the live inference
    pipeline. Load it with:
        import joblib
        model = joblib.load("model/bci_classifier.pkl")
        label = model.predict(feature_vector.reshape(1, -1))[0]
"""

import time

from loader       import load_all_sessions
from preprocessor import preprocess
from features     import build_feature_matrix
from trainer      import train_and_save


def _separator(title: str = "") -> None:
    width = 55
    if title:
        pad   = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print(f"\n{'═' * width}")


def run():
    t_start = time.time()

    _separator("BCI TRAINING PIPELINE")

    # ── 1. Load ──────────────────────────────────────────────────────────
    _separator("1 / 4  LOAD")
    df = load_all_sessions()

    # ── 2. Preprocess ────────────────────────────────────────────────────
    _separator("2 / 4  PREPROCESS")
    epochs, labels, session_ids = preprocess(df)

    # ── 3. Feature extraction ────────────────────────────────────────────
    _separator("3 / 4  FEATURES")
    X = build_feature_matrix(epochs)

    # ── 4. Train & evaluate ──────────────────────────────────────────────
    _separator("4 / 4  TRAIN")
    model = train_and_save(X, labels, session_ids)

    elapsed = time.time() - t_start
    _separator("DONE")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Model ready at: model/bci_classifier.pkl")
    _separator()

    return model


if __name__ == "__main__":
    run()

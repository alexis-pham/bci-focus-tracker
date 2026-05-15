"""
trainer.py — Model training, LOSO cross-validation, evaluation, and saving.

Two models are evaluated:
  - SVM (RBF kernel): handles nonlinear relationships, robust on small data
  - LDA: fast linear baseline — if it matches SVM, prefer it for inference speed

Cross-validation strategy: Leave-One-Session-Out (LOSO)
  - Each fold holds out one complete recording session as the test set
  - Training set = all other sessions
  - This is the most honest estimate of live performance because the model
    has never seen any data from the held-out time period
  - Respects temporal and session boundaries — no data leakage

After evaluation, the winning model is retrained on ALL data and saved as
a single scikit-learn Pipeline object. The live inference pipeline loads
this object and calls .predict() on a raw feature vector — scaling is
applied automatically inside the pipeline.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import MODEL_PATH, MODEL_DIR, LABEL_NAMES


# ---------------------------------------------------------------------------
# Pipeline constructors
# ---------------------------------------------------------------------------

def build_svm() -> Pipeline:
    """
    StandardScaler → SVM (RBF kernel).

    C=1.0 and gamma='scale' are sensible defaults; grid search can refine
    these if LOSO accuracy is unsatisfying.
    """
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",  # handles mild class imbalance
            random_state=42,
        )),
    ])


def build_lda() -> Pipeline:
    """
    StandardScaler → LDA.

    LDA is fast and interpretable. If it performs comparably to SVM it is
    the better choice for the live pipeline due to lower inference latency.
    """
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LinearDiscriminantAnalysis()),
    ])


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def loso_evaluate(
    X:           np.ndarray,
    y:           np.ndarray,
    session_ids: np.ndarray,
    pipeline:    Pipeline,
    model_name:  str,
) -> Dict:
    """
    Run Leave-One-Session-Out cross-validation and collect metrics.

    Args:
        X:           (n_epochs, n_features)
        y:           (n_epochs,) integer labels
        session_ids: (n_epochs,) session identifier per epoch
        pipeline:    untrained scikit-learn Pipeline
        model_name:  display name for logging

    Returns:
        dict with keys: model_name, pipeline (last fitted fold),
        mean_accuracy, std_accuracy, fold_accuracies, all_true, all_pred
    """
    sessions = np.unique(session_ids)
    n_folds  = len(sessions)

    fold_accuracies: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    print(f"\n  ── {model_name}  (LOSO, {n_folds} folds) ──")

    for fold_i, held_out in enumerate(sessions):
        train_mask = session_ids != held_out
        test_mask  = session_ids == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        class_counts = {LABEL_NAMES[k]: int((y_test == k).sum()) for k in LABEL_NAMES}
        print(
            f"    Fold {fold_i + 1:02d}/{n_folds}  "
            f"[{held_out}]  "
            f"acc={acc:.3f}  "
            f"n_test={len(y_test)} {class_counts}"
        )

    mean_acc = float(np.mean(fold_accuracies))
    std_acc  = float(np.std(fold_accuracies))

    print(f"\n  Mean accuracy : {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"\n  Classification report (aggregated across all folds):")
    print(
        classification_report(
            all_true, all_pred,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            digits=3,
        )
    )
    cm = confusion_matrix(all_true, all_pred)
    print(f"  Confusion matrix  [rows=true, cols=pred]:")
    print(f"                  focused  distracted")
    print(f"    focused     {cm[0, 0]:>7}  {cm[0, 1]:>10}")
    print(f"    distracted  {cm[1, 0]:>7}  {cm[1, 1]:>10}")

    return {
        "model_name":      model_name,
        "pipeline":        pipeline,
        "mean_accuracy":   mean_acc,
        "std_accuracy":    std_acc,
        "fold_accuracies": fold_accuracies,
        "all_true":        all_true,
        "all_pred":        all_pred,
    }


# ---------------------------------------------------------------------------
# Model selection, final training, and saving
# ---------------------------------------------------------------------------

def train_and_save(
    X:           np.ndarray,
    y:           np.ndarray,
    session_ids: np.ndarray,
) -> Pipeline:
    """
    Evaluate SVM and LDA via LOSO, select the better model,
    retrain it on the full dataset, and save it to disk.

    Args:
        X:           (n_epochs, n_features)
        y:           (n_epochs,) integer labels
        session_ids: (n_epochs,) session identifiers

    Returns:
        The final trained Pipeline, ready for inference.
    """
    candidates = [
        ("SVM", build_svm()),
        ("LDA", build_lda()),
    ]

    results: Dict[str, Dict] = {}
    for name, pipeline in candidates:
        results[name] = loso_evaluate(X, y, session_ids, pipeline, name)

    # ── Model selection ──────────────────────────────────────────────────
    best_name   = max(results, key=lambda k: results[k]["mean_accuracy"])
    best_result = results[best_name]
    other_name  = [n for n in results if n != best_name][0]

    print(f"\n{'─' * 55}")
    print(f"  Model selection:")
    for name in [best_name, other_name]:
        r   = results[name]
        tag = "  ← SELECTED" if name == best_name else ""
        print(
            f"    {name:5s}  mean_acc={r['mean_accuracy']:.3f} "
            f"± {r['std_accuracy']:.3f}{tag}"
        )
    print(f"{'─' * 55}")

    # ── Retrain on all data ───────────────────────────────────────────────
    # Build a fresh instance so the final model is not biased by the last
    # LOSO fold's fitted state.
    if best_name == "SVM":
        final_pipeline = build_svm()
    else:
        final_pipeline = build_lda()

    print(f"\n  Retraining {best_name} on full dataset ({len(X)} epochs)...")
    final_pipeline.fit(X, y)
    print("  Retraining complete.")

    # ── Save ─────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"  Saved → {MODEL_PATH}  ({size_kb:.1f} KB)")

    return final_pipeline

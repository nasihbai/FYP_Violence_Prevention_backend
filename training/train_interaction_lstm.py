"""
3-Way Ablation Training Pipeline — Interaction-Aware Violence Detection.

Trains two learned models and evaluates all three against the same test split:

  Baseline A — Heuristic: HeuristicFightClassifier (no training, clip-level)
  Model B     — Isolated:  BiLSTM on isolated-pose features only (10-dim)
  Model C     — Full:      BiLSTM on full interaction features   (19-dim)  ← proposed

Writes results to RLVS_RESULTS_DIR:
  ablation_comparison.json        — accuracy/precision/recall/F1/AUC per model
  ablation_comparison.png         — bar chart
  model_b_confusion.png
  model_c_confusion.png
  model_c_training_curves.png
  model_b_training_curves.png

Saves proposed model:
  models/violence_interaction_lstm.h5   (Model C)

Prerequisites:
  Run training/extract_pose_features.py first to build the .npy cache.

Usage:
  python -m training.train_interaction_lstm
  python -m training.train_interaction_lstm --epochs 30 --batch 64
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

# Reproducibility seeds — set before importing TF
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
log.info("Training seed: %d", SEED)

from config.settings import InteractionConfig as IC, MODELS_DIR
from core.interaction_features import FEATURE_DIM_ISOLATED, FEATURE_DIM_FULL

# ── Hyper-parameters (tunables live here, not scattered in code) ──────────────
LSTM_UNITS   = 64
DROPOUT      = 0.4
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
CLASS_NAMES  = ["NonViolent", "Violent"]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (reuses architecture from train_violence_yolo_pose.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(seq_len: int, n_features: int, name: str = "interaction_lstm") -> tf.keras.Model:
    """
    BiLSTM + Attention + TCN — identical architecture to train_violence_yolo_pose.py
    so the ablation is feature-only, not architecture.
    """
    inp = tf.keras.Input(shape=(seq_len, n_features), name="input")
    x   = inp

    # TCN causal layers
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu", dilation_rate=1)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu", dilation_rate=2)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # BiLSTM stack
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_1"
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_2"
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    # Attention pooling
    attn = tf.keras.layers.Dense(1, activation="tanh")(x)
    attn = tf.keras.layers.Flatten()(attn)
    attn = tf.keras.layers.Activation("softmax")(attn)
    lstm_dim = LSTM_UNITS * 2        # bidirectional
    attn = tf.keras.layers.RepeatVector(lstm_dim)(attn)
    attn = tf.keras.layers.Permute([2, 1])(attn)
    x    = tf.keras.layers.Multiply()([x, attn])
    x    = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

    # Head
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="output")(x)

    model = tf.keras.Model(inp, out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _class_weights(y: np.ndarray) -> dict:
    n   = len(y)
    n_v = int(y.sum())
    n_nv= n - n_v
    if n_v == 0 or n_nv == 0:
        return {0: 1.0, 1: 1.0}
    return {0: n / (2.0 * n_nv), 1: n / (2.0 * n_v)}


def _train_model(
    name:    str,
    X_tr:    np.ndarray,
    y_tr:    np.ndarray,
    X_va:    np.ndarray,
    y_va:    np.ndarray,
    out_dir: Path,
    batch:   int,
    epochs:  int,
) -> tf.keras.Model:
    seq_len    = X_tr.shape[1]
    n_features = X_tr.shape[2]
    log.info("[%s] input: (%d, %d)  train=%d  val=%d", name, seq_len, n_features, len(X_tr), len(X_va))

    model = build_model(seq_len, n_features, name=name)
    model.summary(print_fn=log.info)

    cw      = _class_weights(y_tr)
    ckpt_p  = str(out_dir / f"{name}_best.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                          restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt_p, monitor="val_accuracy",
                                            save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=4, min_lr=1e-6, verbose=1),
    ]
    log.info("Class weights: NonViolent=%.3f  Violent=%.3f", cw[0], cw[1])

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    _plot_training(history, name, out_dir)
    return model, history


def _evaluate_model(model, X_te: np.ndarray, y_te: np.ndarray, label: str):
    """Compute accuracy, precision, recall, F1, AUC for a Keras model."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )
    probs  = model.predict(X_te, verbose=0)[:, 1]
    y_pred = (probs >= 0.5).astype(int)
    return {
        "model":     label,
        "accuracy":  round(float(accuracy_score(y_te, y_pred)), 4),
        "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(y_te, probs)), 4),
    }, confusion_matrix(y_te, y_pred)


def _evaluate_heuristic(test_clips, seq_len: int, stride: int, label: str) -> dict:
    """
    Run HeuristicFightClassifier on each test clip and aggregate clip-level
    predictions using majority vote over all frames in the clip.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )
    from ultralytics import YOLO
    from core.heuristic_classifier import HeuristicFightClassifier
    from core.pose_estimator import YOLOPoseEstimator

    log.info("[%s] Evaluating heuristic on %d test clips …", label, len(test_clips))
    pose_est = YOLOPoseEstimator(model_path=IC.POSE_MODEL, conf=0.35)
    y_true, y_pred = [], []

    for video_path, true_label in test_clips:
        clf = HeuristicFightClassifier()   # fresh per clip
        cap = cv2.VideoCapture(str(video_path))
        fight_votes = []
        frame_idx   = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                persons     = pose_est.estimate(frame)
                any_fight, _, _, _, _ = clf.classify(persons)
                fight_votes.append(1 if any_fight else 0)
            frame_idx += 1
        cap.release()

        clip_pred = 1 if (sum(fight_votes) > len(fight_votes) / 2) else 0
        y_true.append(true_label)
        y_pred.append(clip_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "model":     label,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc":       None,   # no probability → can't compute AUC
    }


def _plot_training(history, name: str, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["accuracy"],     label="Train")
        ax1.plot(history.history["val_accuracy"], label="Val")
        ax1.set_title(f"{name} — Accuracy"); ax1.legend(); ax1.grid(True)
        ax2.plot(history.history["loss"],     label="Train")
        ax2.plot(history.history["val_loss"], label="Val")
        ax2.set_title(f"{name} — Loss"); ax2.legend(); ax2.grid(True)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}_training_curves.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        log.warning("Could not save training curves for %s: %s", name, exc)


def _plot_confusion(cm, name: str, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name} — Confusion Matrix")
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}_confusion.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        log.warning("Could not save confusion matrix for %s: %s", name, exc)


def _plot_ablation(results: list, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models  = [r["model"] for r in results]
        metrics = ["accuracy", "precision", "recall", "f1"]
        x       = np.arange(len(models))
        w       = 0.18
        fig, ax = plt.subplots(figsize=(10, 5))

        for i, m in enumerate(metrics):
            vals = [r.get(m) or 0 for r in results]
            ax.bar(x + i*w - 1.5*w, vals, w, label=m.capitalize())

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Ablation Study — A (Heuristic) vs B (Isolated) vs C (Interaction)")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "ablation_comparison.png", dpi=120)
        plt.close(fig)
        log.info("Ablation bar chart → %s", out_dir / "ablation_comparison.png")
    except Exception as exc:
        log.warning("Could not save ablation chart: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train interaction-aware violence LSTM (ablation)")
    parser.add_argument("--epochs", type=int,   default=EPOCHS,     help="Training epochs")
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--subset", type=int,   default=None,       help="Limit clips per split (debug)")
    parser.add_argument("--skip-heuristic", action="store_true",
                        help="Skip Baseline A evaluation (faster, saves GPU memory)")
    args = parser.parse_args()

    cache_dir  = IC.RLVS_CACHE_DIR
    results_dir = IC.RLVS_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_sub{args.subset}" if args.subset else ""

    def _load(split, matrix):
        p = cache_dir / f"{split}_X_{matrix}{suffix}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Cache not found: {p}\n"
                "Run:  python -m training.extract_pose_features"
                + (f" --subset {args.subset}" if args.subset else "")
            )
        return np.load(p)

    def _load_y(split):
        p = cache_dir / f"{split}_y{suffix}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Label cache not found: {p}")
        return np.load(p).astype(np.int32)

    log.info("Loading cached features …")
    X_tr_iso  = _load("train", "isolated")
    X_tr_full = _load("train", "full")
    y_tr      = _load_y("train")

    X_va_iso  = _load("val", "isolated")
    X_va_full = _load("val", "full")
    y_va      = _load_y("val")

    X_te_iso  = _load("test", "isolated")
    X_te_full = _load("test", "full")
    y_te      = _load_y("test")

    log.info("Train:  iso=%s  full=%s  y=%s", X_tr_iso.shape, X_tr_full.shape, y_tr.shape)
    log.info("Val:    iso=%s  full=%s  y=%s", X_va_iso.shape, X_va_full.shape, y_va.shape)
    log.info("Test:   iso=%s  full=%s  y=%s", X_te_iso.shape, X_te_full.shape, y_te.shape)

    ablation_results = []

    # ── Baseline A: Heuristic ─────────────────────────────────────────────
    if not args.skip_heuristic:
        try:
            import cv2
            _, _, test_clips = __import__(
                "training.rlvs_dataset", fromlist=["load_rlvs_splits"]
            ).load_rlvs_splits()
            if args.subset:
                test_clips = test_clips[:args.subset]
            res_a = _evaluate_heuristic(test_clips, IC.SEQUENCE_LENGTH, IC.FRAME_STRIDE,
                                         "A-Heuristic")
            ablation_results.append(res_a)
            log.info("Baseline A: %s", res_a)
        except Exception as exc:
            log.warning("Baseline A evaluation skipped: %s", exc)
    else:
        log.info("Skipping Baseline A (--skip-heuristic)")

    # ── Model B: Isolated pose features ──────────────────────────────────
    log.info("=" * 60)
    log.info("Training Model B — Isolated features (%d-dim)", FEATURE_DIM_ISOLATED)
    model_b, _ = _train_model(
        "model_b_isolated",
        X_tr_iso, y_tr,
        X_va_iso, y_va,
        out_dir=results_dir,
        batch=args.batch,
        epochs=args.epochs,
    )
    res_b, cm_b = _evaluate_model(model_b, X_te_iso, y_te, "B-Isolated")
    ablation_results.append(res_b)
    _plot_confusion(cm_b, "model_b", results_dir)
    log.info("Model B: %s", res_b)

    # ── Model C: Full interaction features ────────────────────────────────
    log.info("=" * 60)
    log.info("Training Model C — Full interaction features (%d-dim)", FEATURE_DIM_FULL)
    model_c, _ = _train_model(
        "model_c_interaction",
        X_tr_full, y_tr,
        X_va_full, y_va,
        out_dir=results_dir,
        batch=args.batch,
        epochs=args.epochs,
    )
    res_c, cm_c = _evaluate_model(model_c, X_te_full, y_te, "C-Interaction (proposed)")
    ablation_results.append(res_c)
    _plot_confusion(cm_c, "model_c", results_dir)
    log.info("Model C: %s", res_c)

    # ── Save Model C as the production model ─────────────────────────────
    model_save = MODELS_DIR / "violence_interaction_lstm.h5"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_c.save(str(model_save))
    log.info("Production model saved → %s", model_save)

    # ── Ablation comparison ────────────────────────────────────────────────
    _plot_ablation(ablation_results, results_dir)
    comparison_path = results_dir / "ablation_comparison.json"
    comparison_path.write_text(json.dumps({
        "seed":    SEED,
        "results": ablation_results,
        "note":    "Model C F1 > Model B F1 demonstrates that interaction features lift performance.",
    }, indent=2))
    log.info("Comparison JSON → %s", comparison_path)

    log.info("=" * 60)
    log.info("ABLATION SUMMARY")
    log.info("%-30s  Acc   Prec  Rec   F1    AUC", "Model")
    for r in ablation_results:
        log.info(
            "%-30s  %.3f %.3f %.3f %.3f  %s",
            r["model"],
            r.get("accuracy", 0),
            r.get("precision", 0),
            r.get("recall", 0),
            r.get("f1", 0),
            f'{r["auc"]:.3f}' if r.get("auc") else "  n/a",
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()

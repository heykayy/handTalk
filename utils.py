"""
utils.py – ISL Translator: dataset loading, label-map I/O, training-history plot
==================================================================================

"""

import os
import json

import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving plots
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 16          # global default; train.ipynb overrides via batch_size= arg
SEED = 123               # MUST match between the "training" and "validation" subset
                         # calls below, or the two splits won't be complementary.

from app_paths import app_root

BASE_DIR = app_root()
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "word", "isl_final_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "word", "label_map.json")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def get_generators_from_directory(dataset_path, batch_size: int = BATCH_SIZE,
                                   img_size: int = IMG_SIZE):
    """
    Build train/val tf.data.Dataset objects from an image directory
    (one sub-folder per class), plus aligned numpy label arrays.

    Returns
    -------
    train_ds, val_ds, label_map, class_names, num_classes, train_labels, val_labels
    """
    # 1) Load RAW datasets first — class_names is only available here, before
    #    any .map()/.cache()/.prefetch() transform is applied.
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )

    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,   # <- critical: keeps validation order fixed & reproducible
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    if class_names != val_ds_raw.class_names:
        raise RuntimeError(
            "Train/val class_names disagree — the dataset folders discovered "
            f"for training ({class_names}) don't match validation "
            f"({val_ds_raw.class_names}). Check for empty/extra class folders."
        )

    label_map = {i: name for i, name in enumerate(class_names)}

    # 2) Extract label arrays ONCE, directly from these raw (pre-shuffle-drift)
    #    datasets, in the exact iteration order that will be reused later.
    train_labels = np.concatenate([y.numpy() for _, y in train_ds_raw], axis=0)
    val_labels = np.concatenate([y.numpy() for _, y in val_ds_raw], axis=0)

    print("[utils] Validation samples per class (sanity check):")
    for i, name in enumerate(class_names):
        count = int((val_labels == i).sum())
        flag = "  <-- ZERO SAMPLES: check this class's folder!" if count == 0 else ""
        print(f"    {name:<6} {count:>5}{flag}")

    # 3) Normalise to [0,1] — word_model.py's build_model() expects [0,1] input
    #    (no preprocess_input applied inside the model).
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds_raw.map(lambda x, y: (normalize(x), y),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds_raw.map(lambda x, y: (normalize(x), y),
                             num_parallel_calls=tf.data.AUTOTUNE)

    # Training set: caching + shuffling + prefetch is safe here because we
    # already captured train_labels above and train order doesn't need to be
    # reproduced exactly again.
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(tf.data.AUTOTUNE)

    # Validation set: cache + prefetch WITHOUT shuffle, so every subsequent
    # pass (class weighting already done above, model.predict(), per-class
    # accuracy) sees images in the same order as val_labels.
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_map, class_names, num_classes, train_labels, val_labels


# ─────────────────────────────────────────────────────────────────────────────
# Label map I/O
# ─────────────────────────────────────────────────────────────────────────────
def save_label_map(label_map: dict, path: str = LABEL_MAP_PATH) -> None:
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"[utils] Saved label map -> {path}")


def load_label_map(path: str = LABEL_MAP_PATH) -> dict:
    """
    Load label_map.json and return it keyed by INT, not str.

    json.load() always returns string keys ("0", "1", ... "34") because JSON
    object keys are strings by definition -- that's correct for the file on
    disk. But predict.py looks classes up by the model's raw argmax output,
    which is a Python int (`idx = int(np.argmax(preds))`), via
    `letter_label_map[idx]` / `idx not in letter_label_map`. A dict keyed by
    "4" never matches a lookup for 4, so every single prediction failed that
    check and fell through to the "[WARN] ... not in label_map" branch --
    this is why WORD mode showed nothing but warnings while SENTENCE mode
    (which doesn't go through this map) worked fine. Converting keys to int
    here, once, fixes every caller instead of patching each lookup site.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Training history plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history, save_path: str = None) -> None:
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "word", "plots", "training_history.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"[Utils] Training plot saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Real-time prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Track frames per second using exponential moving average."""
    
    def __init__(self, avg_over: int = 30):
        """
        Initialize FPS counter.
        
        Parameters
        ----------
        avg_over : int
            Number of frames to average over for EMA calculation.
        """
        self.avg_over = avg_over
        self.last_time = None
        self.fps = 0.0
    
    def tick(self, current_time: float) -> float:
        """
        Update FPS with current timestamp.
        
        Parameters
        ----------
        current_time : float
            Current time in seconds (e.g., from time.time()).
        
        Returns
        -------
        float
            Current estimated FPS.
        """
        if self.last_time is None:
            self.last_time = current_time
            return 0.0
        
        dt = current_time - self.last_time
        if dt > 0:
            frame_fps = 1.0 / dt
            # Exponential moving average
            alpha = 2.0 / (self.avg_over + 1)
            self.fps = alpha * frame_fps + (1 - alpha) * self.fps
        
        self.last_time = current_time
        return self.fps


class PredictionSmoother:
    """Smooth noisy predictions by maintaining a window of recent predictions."""
    
    def __init__(self, window_size: int = 3):
        """
        Initialize prediction smoother.
        
        Parameters
        ----------
        window_size : int
            Size of the sliding window for majority voting.
        """
        self.window_size = window_size
        self.window = []
    
    def update(self, prediction: str) -> str:
        """
        Update with new prediction and return smoothed result.
        
        Parameters
        ----------
        prediction : str
            Current frame's predicted label.
        
        Returns
        -------
        str
            Most common label in the window (majority voting).
        """
        self.window.append(prediction)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        # Majority voting — return most common label in window
        if self.window:
            from collections import Counter
            counts = Counter(self.window)
            return counts.most_common(1)[0][0]
        return prediction
    
    def reset(self) -> None:
        """Clear the prediction window."""
        self.window = []


def draw_overlay(canvas: np.ndarray, label: str, conf: float,
                 fps: float, x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Draw hand detection ROI box and prediction overlay on canvas.
    
    Parameters
    ----------
    canvas : np.ndarray
        Image to draw on (BGR format).
    label : str
        Predicted letter label.
    conf : float
        Prediction confidence [0, 1].
    fps : float
        Frames per second counter.
    x1, y1, x2, y2 : int
        Bounding box coordinates (pixel coordinates).
    """
    if x2 <= x1 or y2 <= y1:
        # No valid ROI
        return
    
    # Draw ROI bounding box — color changes based on confidence
    box_color = (0, 255, 120) if conf >= 0.55 else (100, 100, 200)
    thickness = 2 if conf >= 0.55 else 1
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
    
    # Draw prediction label and confidence near top-left of box
    label_text = f"{label.upper()}  {conf*100:.0f}%"
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 
                                 0.85, 2)[0]
    label_x = max(0, x1)
    label_y = max(30, y1 - 8)
    
    # Draw background rectangle for text
    cv2.rectangle(canvas, 
                  (label_x - 4, label_y - text_size[1] - 6),
                  (label_x + text_size[0] + 4, label_y + 4),
                  (20, 20, 30), -1)
    
    # Draw text
    text_color = (0, 255, 120) if conf >= 0.55 else (100, 180, 255)
    cv2.putText(canvas, label_text, 
                (label_x, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, text_color, 2, cv2.LINE_AA)
    
    # Draw FPS in top-left corner
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)

"""
word_model.py - ISL Translator Model Architecture
Defines MobileNetV2-based transfer learning model optimized for real-time inference.

Second pass of tuning, after confirming (via the fixed utils.py) that the
dataset itself is properly balanced (900/class, one class at 990) and that
validation samples now land in sane per-class buckets (~150-210 each out of
6,318). Changes in this revision:

"""

import tensorflow as tf
from keras import layers, models, applications
from keras.optimizers import Adam
from keras.regularizers import l2


@tf.keras.utils.register_keras_serializable(package="isl")
class SparseCategoricalCrossentropyWithSmoothing(tf.keras.losses.Loss):
    """
    Sparse-label cross-entropy with label smoothing that works on any
    TF/Keras version. Internally one-hots y_true and delegates to
    `categorical_crossentropy`, whose `label_smoothing` argument has been
    supported for a long time (unlike SparseCategoricalCrossentropy's).
    """

    def __init__(self, num_classes: int, label_smoothing: float = 0.05,
                 name: str = "sparse_categorical_crossentropy_ls", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes)
        return tf.keras.losses.categorical_crossentropy(
            y_true_oh, y_pred, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "label_smoothing": self.label_smoothing,
        })
        return config


def _make_loss(num_classes: int, label_smoothing: float):
    if label_smoothing <= 0:
        return "sparse_categorical_crossentropy"
    return SparseCategoricalCrossentropyWithSmoothing(
        num_classes=num_classes, label_smoothing=label_smoothing
    )


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3),
                learning_rate: float = 1e-3, label_smoothing: float = 0.05,
                use_augmentation: bool = True):
    """
    Build a MobileNetV2-based transfer learning model for ISL gesture classification.

    Args:
        num_classes      : Number of gesture classes to classify (35 for
                            digits 1-9 + letters A-Z).
        input_shape       : Input image dimensions (H, W, C).
        learning_rate     : Adam optimizer learning rate.
        label_smoothing   : Smoothing factor for the loss (0.0 disables it).
        use_augmentation  : Whether to include in-graph augmentation layers.
                            These layers are only active when the model is
                            called with training=True (i.e. during .fit()),
                            and are automatically skipped during
                            .predict()/.evaluate() and at inference time.

    Returns:
        (compiled Keras model, base_model reference)
    """
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # ── Phase 1: Freeze all base layers for initial training ──────────────────
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    if use_augmentation:
        # Dialed back vs. the previous revision: ISL numerals/letters are
        # often distinguished by small handshape differences, so aggressive
        # geometric augmentation risks blurring classes into each other
        # rather than just adding robustness. No flips — ISL hand-shapes are
        # not mirror-symmetric, and mirroring silently swaps some classes'
        # visual identity while keeping the original label.
        x = layers.RandomRotation(0.035, fill_mode="nearest")(x)   # ~±12.6°
        x = layers.RandomZoom(0.06, fill_mode="nearest")(x)
        x = layers.RandomTranslation(0.04, 0.04, fill_mode="nearest")(x)
        x = layers.RandomBrightness(0.15)(x)
        x = layers.RandomContrast(0.15)(x)

    # NOTE: preprocessing is done by the data generator (rescale=1/255)
    # and by preprocess_roi at inference. Do NOT apply preprocess_input here
    # — doing so on already-normalised [0,1] inputs would produce [0,0.008],
    # which destroys MobileNetV2 features entirely (confirmed root cause of
    # model collapse to single-class prediction).
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(384, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ISL_MobileNetV2")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=_make_loss(num_classes, label_smoothing),
        metrics=["accuracy"]
    )

    return model, base_model


def unfreeze_top_layers(model: tf.keras.Model, base_model: tf.keras.Model,
                        num_classes: int,
                        num_layers_to_unfreeze: int = 45,
                        learning_rate: float = 5e-5,
                        label_smoothing: float = 0.05) -> tf.keras.Model:
    """
    Fine-tune by unfreezing the top N layers of the base model (Phase 2).

    Args:
        model                 : Full compiled model.
        base_model            : MobileNetV2 base model reference.
        num_classes           : Number of classes (needed to rebuild the loss).
        num_layers_to_unfreeze: Number of top base layers to unfreeze.
                                Raised from 30 -> 45: 31.5k balanced training
                                images can support tuning deeper features
                                without overfitting.
        learning_rate         : Lower LR to prevent catastrophic forgetting.
        label_smoothing       : Smoothing factor for the loss (0.0 disables it).

    Returns:
        Re-compiled model with some base layers trainable.
    """
    base_model.trainable = True

    # Keep BatchNorm layers inside the base model frozen even when unfreezing
    # top conv layers — updating BN running stats on a smaller fine-tuning
    # batch size can destabilise the pretrained features.
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=_make_loss(num_classes, label_smoothing),
        metrics=["accuracy"]
    )

    print(f"[Model] Unfroze top {num_layers_to_unfreeze} base layers for fine-tuning "
          f"(BatchNorm layers kept frozen).")
    return model


def print_model_summary(model: tf.keras.Model) -> None:
    """Print model summary with trainable/non-trainable parameter counts."""
    model.summary()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    print(f"\n[Model] Trainable params     : {trainable:,}")
    print(f"[Model] Non-trainable params : {non_trainable:,}")

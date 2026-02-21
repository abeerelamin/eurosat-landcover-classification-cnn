import warnings
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    Multiply,
    Reshape,
    Concatenate,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable

from data import IMG_SIZE, NUM_CLASSES

warnings.filterwarnings(
    "ignore",
    message="Gradients do not exist for variables",
    category=UserWarning,
)

# Disable XLA so GPU kernels don't break on projective transform
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass


# ==== Custom STN layer ====
@register_keras_serializable(package="custom")
class SimpleSpatialTransformer(tf.keras.layers.Layer):
    """
    Minimal affine Spatial Transformer using TF-only ops.
    Uses a tiny localization net to predict 6 affine params (a0..a5),
    applies tf.raw_ops.ImageProjectiveTransformV3 on CPU.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loc_conv1 = Conv2D(8, 7, activation="relu")
        self.pool1 = MaxPooling2D()
        self.loc_conv2 = Conv2D(10, 5, activation="relu")
        self.pool2 = MaxPooling2D()
        self.flatten = Flatten()
        self.fc = Dense(
            6,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer=tf.constant_initializer(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ),
        )

    def call(self, x):
        y = self.loc_conv1(x)
        y = self.pool1(y)
        y = self.loc_conv2(y)
        y = self.pool2(y)
        y = self.flatten(y)

        theta = self.fc(y)  # (B, 6)
        theta = tf.cast(theta, x.dtype)

        # Add last 2 params for projective transform (set to zero)
        transforms = tf.concat(
            [
                theta,
                tf.zeros((tf.shape(theta)[0], 2), dtype=theta.dtype),
            ],
            axis=1,  # (B, 8)
        )

        h = tf.shape(x)[1]
        w = tf.shape(x)[2]

        # Run on CPU to avoid missing GPU kernels
        with tf.device("/CPU:0"):
            out = tf.raw_ops.ImageProjectiveTransformV3(
                images=x,
                transforms=transforms,
                output_shape=tf.stack([h, w]),
                interpolation="BILINEAR",
                fill_value=0.0,
            )
        return out

    def get_config(self):
        return super().get_config()


# ==== Attention blocks ====
def channel_attention(x, reduction=8):
    c = x.shape[-1]
    gap = GlobalAveragePooling2D()(x)  # (B, C)
    fc1 = Dense(max(c // reduction, 1), activation="relu")(gap)
    fc2 = Dense(c, activation="sigmoid")(fc1)
    scale = Reshape((1, 1, c))(fc2)
    return Multiply()([x, scale])


def spatial_attention(x):
    # Keras-3 friendly: wrap TF ops with Lambda
    avg_pool = Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True)
    )(x)
    max_pool = Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True)
    )(x)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attn = Conv2D(
        1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
    )(concat)
    return Multiply()([x, attn])


# ==== Base CNN ====
def build_base_cnn(num_classes: int = NUM_CLASSES):
    model = Sequential(
        [
            Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            Conv2D(32, 3, activation="relu"),
            MaxPooling2D(),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==== CNN with attention ====
def build_attention_cnn(num_classes: int = NUM_CLASSES):
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(32, 3, activation="relu")(inp)
    x = MaxPooling2D()(x)
    x = channel_attention(x)
    x = spatial_attention(x)

    x = Conv2D(64, 3, activation="relu")(x)
    x = MaxPooling2D()(x)
    x = channel_attention(x)
    x = spatial_attention(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==== CNN with STN ====
def build_stn_cnn(num_classes: int = NUM_CLASSES):
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = SimpleSpatialTransformer()(inp)
    x = Conv2D(32, 3, activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,  # important for Colab
    )
    return model

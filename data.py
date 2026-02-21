import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 42
tf.keras.utils.set_random_seed(SEED)

IMG_SIZE = 64
BATCH = 128
EPOCHS = 20  # set to 10+ for stronger results
AUTOTUNE = tf.data.AUTOTUNE

# ==== EuroSAT (RGB) via TFDS ====
(ds_train, ds_val), ds_info = tfds.load(
    "eurosat/rgb",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
)

NUM_CLASSES = ds_info.features["label"].num_classes
CLASS_NAMES = ds_info.features["label"].names


def preprocess(x, y):
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    y = tf.one_hot(y, NUM_CLASSES)
    return x, y


augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
    ]
)

train_ds = (
    ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (augment(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .shuffle(2048, seed=SEED)
    .batch(BATCH)
    .prefetch(AUTOTUNE)
)

val_ds = (
    ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH)
    .prefetch(AUTOTUNE)
)

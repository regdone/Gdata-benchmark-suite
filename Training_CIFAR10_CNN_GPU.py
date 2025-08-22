#!/usr/bin/env python3
# Source/Reference:
# - Keras official CIFAR10 CNN example: https://keras.io/examples/vision/cifar10_cnn/
# Notes:
# - Uses GPU:0, enables memory growth, and does 1 extra warm-up epoch for stable timing.

import os, time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(num_classes=10):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Rescaling(1.0/255)(inputs)
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    epochs = int(os.getenv("EPOCHS", "5"))
    batch_size = int(os.getenv("BATCH", "256"))  # lớn hơn để tận dụng GPU
    extra_warmup = int(os.getenv("WARMUP_EPOCHS", "1"))

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "No GPU found. Check driver/CUDA/TensorFlow build."
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(gpus[0], "GPU")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    model = build_model()

    # Warm-up
    if extra_warmup > 0:
        print(f"Warm-up for {extra_warmup} epoch(s)...")
        model.fit(x_train[:8192], y_train[:8192], epochs=extra_warmup,
                  batch_size=batch_size, verbose=0)

    t0 = time.time()
    hist = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    dt = time.time() - t0
    imgs = len(x_train) * epochs
    ips = imgs / dt
    print(f"Training finished in {dt:.2f} s | throughput ~ {ips:.1f} images/sec (GPU)")
    print("Final val_acc:", hist.history["val_accuracy"][-1])

if __name__ == "__main__":
    main()

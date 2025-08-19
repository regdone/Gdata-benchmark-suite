import tensorflow as tf
import time

print("=== Training MNIST on GPU ===")

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

with tf.device("/GPU:0"):
    start = time.time()
    history = model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=2)
    end = time.time()

print(f"Training finished in {end-start:.2f} seconds (GPU)")


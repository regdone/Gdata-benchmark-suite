import tensorflow as tf
import time

print("=== Real AI Workload: IMDB Sentiment Analysis (GPU) ===")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

with tf.device("/GPU:0"):
    start = time.time()
    model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=2)
    end = time.time()

print(f"Training finished in {end-start:.2f} seconds (GPU)")


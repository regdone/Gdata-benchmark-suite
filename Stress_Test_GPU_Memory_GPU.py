import tensorflow as tf
import time

print("=== Stress Test Memory (GPU) ===")

size = 20000
with tf.device("/GPU:0"):
    start = time.time()
    a = tf.random.uniform([size, size], dtype=tf.float32)
    b = tf.random.uniform([size, size], dtype=tf.float32)
    c = a * b + a
    tf.experimental.numpy.copy(c)
    end = time.time()

print(f"GPU Memory Stress Test Time: {end-start:.2f} seconds")


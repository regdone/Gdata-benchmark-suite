import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import time

print("=== Stress Test Memory (CPU) ===")

size = 20000
with tf.device("/CPU:0"):
    start = time.time()
    a = tf.random.uniform([size, size], dtype=tf.float32)
    b = tf.random.uniform([size, size], dtype=tf.float32)
    c = a * b + a
    tf.experimental.numpy.copy(c)
    end = time.time()

print(f"CPU Memory Stress Test Time: {end-start:.2f} seconds")


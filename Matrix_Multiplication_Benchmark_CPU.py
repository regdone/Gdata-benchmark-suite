import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import time

print("=== Matrix Multiplication Benchmark (CPU) ===")

size = 8000
with tf.device("/CPU:0"):
    a = tf.random.uniform([size, size])
    b = tf.random.uniform([size, size])
    tf.matmul(a, b)  # warmup

    start = time.time()
    c = tf.matmul(a, b)
    tf.experimental.numpy.copy(c)
    end = time.time()

elapsed = end - start
gflops = 2*(size**3) / (elapsed*1e9)
print(f"CPU Time: {elapsed:.4f} s | Throughput: {gflops:.2f} GFLOPS")


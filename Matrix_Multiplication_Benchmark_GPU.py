import tensorflow as tf
import time

print("=== Matrix Multiplication Benchmark (GPU) ===")

size = 8000
with tf.device("/GPU:0"):
    a = tf.random.uniform([size, size])
    b = tf.random.uniform([size, size])
    tf.matmul(a, b)  # warmup

    start = time.time()
    c = tf.matmul(a, b)
    tf.experimental.numpy.copy(c)
    end = time.time()

elapsed = end - start
gflops = 2*(size**3) / (elapsed*1e9)
print(f"GPU Time: {elapsed:.4f} s | Throughput: {gflops:.2f} GFLOPS")


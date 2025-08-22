#!/usr/bin/env python3
# Source/Reference:
# - TensorFlow matmul micro-benchmark patterns used widely in blogs & issues
# - GEMM (matrix multiply) is the de-facto HPC baseline across ML/HPC papers

import os, time
import tensorflow as tf

def gflops(n, secs):
    return (2.0 * (n**3)) / (secs * 1e9)

def main():
    n = int(os.getenv("MATRIX_N", "16384"))  # default lớn hơn CPU để thấy rõ chênh lệch
    runs = int(os.getenv("RUNS", "5"))
    print(f"=== TensorFlow GEMM GPU: N={n}, runs={runs} ===")

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "No GPU found. Check driver/CUDA/TensorFlow build."
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(gpus[0], "GPU")
    except RuntimeError:
        pass

    with tf.device("/GPU:0"):
        a = tf.random.normal((n, n), dtype=tf.float32)
        b = tf.random.normal((n, n), dtype=tf.float32)

        # Warm-up (2 lần để amortize kernel init)
        _ = tf.matmul(a, b); _ = _.numpy()
        _ = tf.matmul(a, b); _ = _.numpy()

        times = []
        for i in range(runs):
            t0 = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()
            dt = time.time() - t0
            times.append(dt)
            print(f"Run {i+1}/{runs}: {dt:.3f} s | {gflops(n, dt):.1f} GFLOP/s")

    avg = sum(times)/len(times)
    print(f"AVG time: {avg:.3f} s | AVG perf: {gflops(n, avg):.1f} GFLOP/s (GPU)")

if __name__ == "__main__":
    main()

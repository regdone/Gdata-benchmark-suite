#!/usr/bin/env python3
# Source/Reference:
# - TensorFlow matmul micro-benchmark patterns used widely in blogs & issues
# - GEMM (matrix multiply) is the de-facto HPC baseline across ML/HPC papers

import os, time, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU before TF import
import tensorflow as tf

def gflops(n, secs):
    # 2*N^3 FLOPs for NxN matmul (A*B +)
    return (2.0 * (n**3)) / (secs * 1e9)

def main():
    n = int(os.getenv("MATRIX_N", "8192"))   # increase to 16384 if you have RAM/Swap
    runs = int(os.getenv("RUNS", "5"))
    print(f"=== TensorFlow GEMM CPU: N={n}, runs={runs} ===")

    a = tf.random.normal((n, n), dtype=tf.float32)
    b = tf.random.normal((n, n), dtype=tf.float32)

    # Warm-up
    _ = tf.matmul(a, b)

    times = []
    for i in range(runs):
        t0 = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()  # ensure compute finished
        dt = time.time() - t0
        times.append(dt)
        print(f"Run {i+1}/{runs}: {dt:.3f} s | {gflops(n, dt):.1f} GFLOP/s")

    avg = sum(times)/len(times)
    print(f"AVG time: {avg:.3f} s | AVG perf: {gflops(n, avg):.1f} GFLOP/s (CPU)")

if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    main()

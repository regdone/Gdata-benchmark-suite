#!/usr/bin/env bash
# Gdata Benchmark Suite
# Script này sẽ chạy toàn bộ 8 benchmark và ghi log ra ./results/
# Yêu cầu: Python >= 3.10 + pip install -r requirements.txt

set -e
mkdir -p results

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="results/benchmark_${timestamp}.log"

echo "=== Gdata Benchmark Suite ===" | tee -a "$logfile"
echo "Start time: $(date)" | tee -a "$logfile"
echo "----------------------------------------" | tee -a "$logfile"

run_py () {
    script=$1
    echo -e "\n>>> Running $script" | tee -a "$logfile"
    echo "----------------------------------------" | tee -a "$logfile"
    { time python3 "$script"; } 2>&1 | tee -a "$logfile"
}

# Matrix Multiplication
MATRIX_N=8192 RUNS=3 run_py Matrix_Multiplication_Benchmark_CPU.py
MATRIX_N=16384 RUNS=3 run_py Matrix_Multiplication_Benchmark_GPU.py

# Training CIFAR10 CNN
EPOCHS=3 BATCH=128 run_py Training_CIFAR10_CNN_CPU.py
EPOCHS=3 BATCH=256 run_py Training_CIFAR10_CNN_GPU.py

# Stress Test Memory
ALLOC_GB=8 CHUNK_MB=256 run_py Stress_Test_GPU_Memory_CPU.py
CHUNK_MB=512 MAX_GB_HINT=80 run_py Stress_Test_GPU_Memory_GPU.py

# NLP Real AI Workload
SAMPLES=2000 BATCH=32 run_py Real_AI_Workload_NLP_CPU.py
SAMPLES=4000 BATCH=64 USE_FP16=1 run_py Real_AI_Workload_NLP_GPU.py

echo -e "\n=== Benchmark finished ===" | tee -a "$logfile"
echo "End time: $(date)" | tee -a "$logfile"
echo "Results saved to $logfile"

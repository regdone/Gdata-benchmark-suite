#!/usr/bin/env bash
# Gdata Benchmark Suite
# Script này sẽ chạy toàn bộ 8 benchmark và ghi log ra ./results/
# Yêu cầu: Python >= 3.10 + pip install -r requirements.txt

#!/bin/bash
set -e

RESULT_FILE="results.md"
echo "# Gdata Benchmark Suite Results" > $RESULT_FILE
echo "| Benchmark | GPU (sec) | CPU (sec) |" >> $RESULT_FILE
echo "|-----------|-----------|-----------|" >> $RESULT_FILE

run_benchmark () {
    NAME=$1
    CPU_SCRIPT=$2
    GPU_SCRIPT=$3

    echo ">>> Running $NAME on CPU..."
    CPU_TIME=$( (time -p python3 $CPU_SCRIPT > /dev/null) 2>&1 | grep real | awk '{print $2}' )
    
    echo ">>> Running $NAME on GPU..."
    GPU_TIME=$( (time -p python3 $GPU_SCRIPT > /dev/null) 2>&1 | grep real | awk '{print $2}' )

    echo "| $NAME | $GPU_TIME | $CPU_TIME |" >> $RESULT_FILE
}

source venv/bin/activate

run_benchmark "Matrix Multiplication" Matrix_Multiplication_Benchmark_CPU.py Matrix_Multiplication_Benchmark_GPU.py
run_benchmark "MNIST Training" Training_MNIST_DeepLearning_CPU.py Training_MNIST_DeepLearning_GPU.py
run_benchmark "Stress Test Memory" Stress_Test_GPU_Memory_CPU.py Stress_Test_GPU_Memory_GPU.py
run_benchmark "Real AI Workload" Real_AI_Workload_CPU.py Real_AI_Workload_GPU.py

echo "Benchmark complete! See $RESULT_FILE"

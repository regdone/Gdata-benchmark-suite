#!/bin/bash
set -e

LOGFILE="benchmark_results.log"
echo "===== GPU/CPU Benchmark Suite =====" > $LOGFILE
echo "Start time: $(date)" >> $LOGFILE
echo "" >> $LOGFILE

run_benchmark () {
    SCRIPT=$1
    echo ">>> Running $SCRIPT" | tee -a $LOGFILE
    echo "----------------------------------------" >> $LOGFILE
    { time python3 $SCRIPT; } >> $LOGFILE 2>&1
    echo "" >> $LOGFILE
}

# List of benchmarks
BENCHMARKS=(
    "Matrix_Multiplication_Benchmark_CPU.py"
    "Matrix_Multiplication_Benchmark_GPU.py"
    "Training_MNIST_DeepLearning_CPU.py"
    "Training_MNIST_DeepLearning_GPU.py"
    "Stress_Test_GPU_Memory_CPU.py"
    "Stress_Test_GPU_Memory_GPU.py"
    "Real_AI_Workload_CPU.py"
    "Real_AI_Workload_GPU.py"
)

for bm in "${BENCHMARKS[@]}"; do
    run_benchmark $bm
done

echo "===== Benchmark completed at $(date) =====" >> $LOGFILE
echo "All results saved to $LOGFILE"

